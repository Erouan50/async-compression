use crate::snappy::{
    crc32c_masked, ChunkType, FrameHeader, MAX_BLOCK_SIZE, MAX_DATA_FRAME_SIZE, STREAM_FRAME,
};
use crate::EncodeV2;
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::mem::MaybeUninit;
use std::{io, mem};

#[derive(Debug)]
pub struct SnappyEncoder {
    state: State,
    in_buf: PartialBuffer<Vec<u8>>,
    out_buf: PartialBuffer<Vec<u8>>,
}

impl Default for SnappyEncoder {
    fn default() -> Self {
        Self {
            state: State::InitStream(PartialBuffer::new(STREAM_FRAME)),
            in_buf: PartialBuffer::new(Vec::new()),
            out_buf: PartialBuffer::new(Vec::new()),
        }
    }
}

impl SnappyEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes one unit of progress from `Buffering`. Emits or stages at most one
    /// block, or accumulates caller input. Returns `false` when more input is
    /// needed before another frame can be produced.
    fn encode_block(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<bool> {
        let output_spare = spare_len(output);
        let in_buf = self.in_buf.get_mut();

        // Check if there is a whole compressible block in the input buffer (and no buffering is
        // in progress: in_buf must be empty)
        if in_buf.is_empty() && input.unwritten().len() >= MAX_BLOCK_SIZE {
            let block = &input.unwritten()[..MAX_BLOCK_SIZE];

            // Check if the output buffer have enough space to fit a compressed or uncompressed
            // frame. If it does, we write the frame directly in the output buffer otherwise we
            // write in the intermediate output buffer and change the state to start the copy of
            // the intermediate output buffer to the output buffer.
            if output_spare >= MAX_DATA_FRAME_SIZE {
                write_frame(block, output)?;
            } else {
                let chunk = compress_into(&mut self.out_buf, block)?;
                // If the chunk is uncompressed, we keep the block in intermediate input buffer
                if matches!(chunk.chunk_type, ChunkType::Uncompressed) {
                    self.in_buf.get_mut().extend_from_slice(block);
                }
                self.state = chunk.pending_state();
            }
            input.advance(MAX_BLOCK_SIZE);
            // We just compressed the block (and maybe even write it!) so we don't need more input
            return Ok(true);
        }

        // Input doesn't bring a whole chunk, so let's accumulate the input in the input
        // intermediate buffer.
        if !input.unwritten().is_empty() && in_buf.capacity() < MAX_BLOCK_SIZE {
            in_buf.reserve(MAX_BLOCK_SIZE - in_buf.len());
        }
        let available = MAX_BLOCK_SIZE - in_buf.len();
        let boundary = available.min(input.unwritten().len());

        in_buf.extend_from_slice(&input.unwritten()[..boundary]);
        input.advance(boundary);

        // If the intermediate buffer is not filled, we need more inputs
        if in_buf.len() < MAX_BLOCK_SIZE {
            return Ok(false);
        }

        // Check when the intermediate input buffer is filled if there is enough space in the
        // output buffer to write directly the frame.
        if output_spare >= MAX_DATA_FRAME_SIZE {
            write_frame(in_buf, output)?;
            clear_for_reuse(&mut self.in_buf);
        } else {
            let chunk = compress_into(&mut self.out_buf, in_buf)?;
            self.state = chunk.pending_state();
        }

        Ok(true)
    }

    /// Drains the pending frame (header + payload) into `output`.
    ///
    /// Returns `true` once the frame is fully written. Both owned buffers are
    /// then cleared for reuse. The caller decides the next state.
    fn write_pending_frame(&mut self, output: &mut WriteBuffer<'_>) -> bool {
        let done = match &mut self.state {
            State::UncompressCopy(header) => write_frame_parts(header, &mut self.in_buf, output),
            State::CompressCopy(header) => write_frame_parts(header, &mut self.out_buf, output),
            _ => unreachable!("write_pending_frame called without a pending frame"),
        };
        if done {
            clear_for_reuse(&mut self.in_buf);
            clear_for_reuse(&mut self.out_buf);
        }
        done
    }
}

fn write_frame_parts(
    header: &mut PartialBuffer<[u8; 8]>,
    payload: &mut PartialBuffer<Vec<u8>>,
    output: &mut WriteBuffer<'_>,
) -> bool {
    if !header.unwritten().is_empty() {
        output.copy_unwritten_from(header);
        if !header.unwritten().is_empty() {
            return false;
        }
    }

    if !payload.unwritten().is_empty() {
        output.copy_unwritten_from(payload);
        payload.unwritten().is_empty()
    } else {
        true
    }
}

fn write_stream_identifier(
    buffer: &mut PartialBuffer<&'static [u8]>,
    output: &mut WriteBuffer<'_>,
) -> bool {
    if !buffer.unwritten().is_empty() {
        output.copy_unwritten_from(buffer);
        if output.has_no_spare_space() {
            return false;
        }
    }
    true
}

struct Chunk {
    chunk_type: ChunkType,
    /// Payload length only (the frame header's length field additionally counts the 4-byte CRC).
    payload_len: usize,
    header: [u8; 8],
}

impl Chunk {
    /// Pending-drain state for this chunk: compressed payloads drain from
    /// `out_buf`, uncompressed payloads drain from `in_buf`.
    fn pending_state(&self) -> State {
        match self.chunk_type {
            ChunkType::Compressed => State::CompressCopy(self.header.into()),
            ChunkType::Uncompressed => State::UncompressCopy(self.header.into()),
            _ => unreachable!(),
        }
    }
}

fn prepare_frame(input: &[u8], output: &mut [MaybeUninit<u8>]) -> std::io::Result<Chunk> {
    let checksum = crc32c_masked(input);

    let mut encoder = snap::raw::Encoder::new();
    let compressed_data = encoder.compress_uninit(input, output)?;

    let (chunk_type, payload_length) = if compressed_data >= input.len() - (input.len() / 8) {
        (ChunkType::Uncompressed, input.len())
    } else {
        (ChunkType::Compressed, compressed_data)
    };

    // We add 4 because the length includes the 4 bytes of the checksum.
    let frame_length = payload_length + 4;
    let header = FrameHeader {
        chunk_type,
        data_frame_length: frame_length as u64,
    };

    let mut raw_chunk_header = [0u8; 8];
    let raw_frame_header: [u8; 4] = header.into();
    let raw_checksum: [u8; 4] = checksum.to_le_bytes();

    raw_chunk_header[0..4].copy_from_slice(&raw_frame_header);
    raw_chunk_header[4..8].copy_from_slice(&raw_checksum);

    Ok(Chunk {
        chunk_type,
        payload_len: payload_length,
        header: raw_chunk_header,
    })
}

fn compress_into(out_buf: &mut PartialBuffer<Vec<u8>>, block: &[u8]) -> std::io::Result<Chunk> {
    clear_for_reuse(out_buf);
    let out = out_buf.get_mut();
    let max_compress_size = snap::raw::max_compress_len(block.len());
    out.reserve(max_compress_size);

    let chunk = prepare_frame(block, &mut out.spare_capacity_mut()[..max_compress_size])?;

    if matches!(chunk.chunk_type, ChunkType::Compressed) {
        // SAFETY: For compressed chunks, `prepare_frame` sets `chunk.payload_len`
        // to the initialized prefix length returned by `compress_uninit`.
        unsafe {
            out.set_len(chunk.payload_len);
        }
    }

    Ok(chunk)
}

fn spare_len(output: &WriteBuffer<'_>) -> usize {
    output.capacity() - output.written_len()
}

fn clear_for_reuse(buf: &mut PartialBuffer<Vec<u8>>) {
    buf.get_mut().clear();
    buf.reset();
}

fn write_frame(input: &[u8], output: &mut WriteBuffer<'_>) -> std::io::Result<()> {
    let required = 8 + snap::raw::max_compress_len(input.len());
    let output_spare = spare_len(output);

    debug_assert!(
        output_spare >= required,
        "write_frame requires {} spare bytes, but only {} remain",
        required,
        output_spare
    );

    // SAFETY: We only write initialized bytes into this slice and never
    // de-initialize bytes already tracked as initialized by `WriteBuffer`.
    let out_buf = unsafe { output.unwritten_mut() };
    let chunk = prepare_frame(input, &mut out_buf[8..required])?;
    write_copy_of_slice(&mut out_buf[..8], &chunk.header);

    // If the chunk is uncompressed, we override the out buffer with the plain input
    if matches!(chunk.chunk_type, ChunkType::Uncompressed) {
        write_copy_of_slice(&mut out_buf[8..8 + input.len()], input);
    }

    unsafe {
        // SAFETY: The header initialized bytes `0..8`. For compressed chunks,
        // `prepare_frame` initialized the following `chunk.payload_len` bytes.
        // For uncompressed chunks, they were initialized by copying `input`.
        output.assume_init_and_advance(8 + chunk.payload_len);
    }
    Ok(())
}

// FIXME: Replace this function with core::mem::maybe_uninit::write_copy_of_slice when the MSRV
//   version of this crate is >= 1.93.0
fn write_copy_of_slice<T>(dest: &mut [MaybeUninit<T>], src: &[T])
where
    T: Copy,
{
    // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
    let uninit_src: &[MaybeUninit<T>] = unsafe { mem::transmute(src) };

    dest.copy_from_slice(uninit_src);
}

#[derive(Debug)]
enum State {
    /// The stream identifier still has bytes to write.
    InitStream(PartialBuffer<&'static [u8]>),
    /// Accumulating caller input until a full block or flush.
    Buffering,
    /// Draining a pending uncompressed frame (the payload is owned by `in_buf`).
    UncompressCopy(PartialBuffer<[u8; 8]>),
    /// Draining a pending compressed frame (the payload is owned by `out_buf`).
    CompressCopy(PartialBuffer<[u8; 8]>),
    /// All pending data emitted; new input returns to `Buffering`.
    Flushed,
}

impl EncodeV2 for SnappyEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> std::io::Result<()> {
        loop {
            match &mut self.state {
                State::InitStream(buffer) => {
                    if !write_stream_identifier(buffer, output) {
                        return Ok(());
                    }
                    self.state = State::Buffering
                }
                State::Buffering => {
                    if !self.encode_block(input, output)? {
                        return Ok(());
                    }
                }
                State::UncompressCopy(_) | State::CompressCopy(_) => {
                    if !self.write_pending_frame(output) {
                        return Ok(());
                    }
                    self.state = State::Buffering;
                }
                State::Flushed => {
                    if input.unwritten().is_empty() {
                        return Ok(());
                    }
                    self.state = State::Buffering
                }
            }
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> std::io::Result<bool> {
        loop {
            match &mut self.state {
                State::InitStream(buffer) => {
                    if !write_stream_identifier(buffer, output) {
                        return Ok(false);
                    }
                    self.state = State::Buffering
                }
                State::Buffering => {
                    let in_buf = self.in_buf.unwritten();
                    let required_capacity = 8 + snap::raw::max_compress_len(in_buf.len());
                    let output_spare = spare_len(output);
                    if output_spare >= required_capacity {
                        // We don't need to use output intermediate buffer, we have space to
                        // compress/copy everything directly to output buffer.
                        write_frame(in_buf, output)?;
                        clear_for_reuse(&mut self.in_buf);
                        self.state = State::Flushed;
                        return Ok(true);
                    }

                    let chunk = compress_into(&mut self.out_buf, in_buf)?;
                    self.state = chunk.pending_state();
                }
                State::UncompressCopy(_) | State::CompressCopy(_) => {
                    if !self.write_pending_frame(output) {
                        return Ok(false);
                    }
                    self.state = State::Flushed;
                }
                State::Flushed => return Ok(true),
            }
        }
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> std::io::Result<bool> {
        self.flush(output)
    }
}

#[cfg(test)]
mod tests {
    use super::write_frame;
    use super::{EncodeV2, SnappyEncoder};
    use crate::snappy::{MAX_BLOCK_SIZE, MAX_DATA_FRAME_SIZE, MAX_FRAME_SIZE, STREAM_FRAME};
    use compression_core::util::{PartialBuffer, WriteBuffer};
    use std::{io::Read, mem::MaybeUninit};

    fn incompressible_block() -> Vec<u8> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;

        (0..MAX_BLOCK_SIZE)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn test_write_frame_compressed_uninitialized_output() {
        let input = vec![0; MAX_BLOCK_SIZE];
        let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);

        write_frame(&input, &mut output).unwrap();

        assert_eq!(output.initialized_len(), output.written_len());

        let mut framed = Vec::with_capacity(STREAM_FRAME.len() + output.written_len());
        framed.extend_from_slice(STREAM_FRAME);
        framed.extend_from_slice(output.written());

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(framed.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, input);
    }

    #[test]
    fn test_write_frame_uncompressed_uninitialized_output() {
        let input = incompressible_block();

        let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);

        write_frame(&input, &mut output).unwrap();

        assert_eq!(output.initialized_len(), output.written_len());
        assert_eq!(output.written()[0], 0x01);

        let mut framed = Vec::with_capacity(STREAM_FRAME.len() + output.written_len());
        framed.extend_from_slice(STREAM_FRAME);
        framed.extend_from_slice(output.written());

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(framed.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_full_block_directly_to_uninitialized_output() {
        let input = vec![0; MAX_BLOCK_SIZE];
        let mut input_buffer = PartialBuffer::new(input.as_slice());

        let mut storage = vec![MaybeUninit::uninit(); MAX_FRAME_SIZE];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);
        let mut encoder = SnappyEncoder::new();

        encoder.encode(&mut input_buffer, &mut output).unwrap();

        assert!(input_buffer.unwritten().is_empty());
        assert_eq!(output.initialized_len(), output.written_len());
        assert_eq!(encoder.in_buf.get_mut().capacity(), 0);
        assert_eq!(encoder.out_buf.get_mut().capacity(), 0);

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(output.written())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_incompressible_full_block_directly_to_uninitialized_output() {
        let input = incompressible_block();
        let mut input_buffer = PartialBuffer::new(input.as_slice());

        let mut storage = vec![MaybeUninit::uninit(); MAX_FRAME_SIZE];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);
        let mut encoder = SnappyEncoder::new();

        encoder.encode(&mut input_buffer, &mut output).unwrap();

        assert!(input_buffer.unwritten().is_empty());
        assert_eq!(output.initialized_len(), output.written_len());
        assert_eq!(output.written()[STREAM_FRAME.len()], 0x01);
        assert_eq!(encoder.in_buf.get_mut().capacity(), 0);
        assert_eq!(encoder.out_buf.get_mut().capacity(), 0);

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(output.written())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_full_block_directly_to_internal_output() {
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let input = vec![0; MAX_BLOCK_SIZE];
            let mut input_buffer = PartialBuffer::new(input.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input_buffer, &mut output).unwrap();

            assert!(input_buffer.unwritten().is_empty());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert_eq!(encoded[STREAM_FRAME.len()], 0x00);
        assert_eq!(encoder.in_buf.get_mut().capacity(), 0);
        assert!(encoder.out_buf.get_mut().capacity() > 0);

        let mut empty_input = PartialBuffer::new(&[][..]);

        {
            let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut empty_input, &mut output).unwrap();

            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, vec![0; MAX_BLOCK_SIZE]);
    }

    #[test]
    fn test_encode_incompressible_full_block_directly_to_internal_output() {
        let expected = incompressible_block();
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let input = expected.clone();
            let mut input_buffer = PartialBuffer::new(input.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input_buffer, &mut output).unwrap();

            assert!(input_buffer.unwritten().is_empty());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert_eq!(encoded[STREAM_FRAME.len()], 0x01);
        assert!(encoder.in_buf.get_mut().capacity() > 0);
        assert!(encoder.out_buf.get_mut().capacity() > 0);

        let mut empty_input = PartialBuffer::new(&[][..]);

        {
            let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut empty_input, &mut output).unwrap();

            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_encode_owned_full_block_directly_to_output() {
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let mut input = PartialBuffer::new(&[][..]);
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();
            encoded.extend_from_slice(output.written());
        }

        {
            let input = vec![0; MAX_BLOCK_SIZE - 1];
            let mut input = PartialBuffer::new(input.as_slice());
            let mut storage: [MaybeUninit<u8>; 0] = [];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            assert_eq!(output.written_len(), 0);
            assert_eq!(encoder.out_buf.get_mut().capacity(), 0);
        }

        {
            let input = [0];
            let mut input = PartialBuffer::new(input.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert!(encoder.in_buf.get_mut().is_empty());
        assert_eq!(encoder.out_buf.get_mut().capacity(), 0);

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, vec![0; MAX_BLOCK_SIZE]);
    }

    #[test]
    fn test_flush_owned_partial_block_directly_to_output() {
        let expected = vec![0; MAX_BLOCK_SIZE / 2];
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let mut input = PartialBuffer::new(expected.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert_eq!(encoder.out_buf.get_mut().capacity(), 0);

        let frame_capacity = 8 + snap::raw::max_compress_len(expected.len());

        {
            let mut storage = vec![MaybeUninit::uninit(); frame_capacity];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            let flushed = encoder.flush(&mut output).unwrap();

            assert!(flushed);
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert!(encoder.in_buf.get_mut().is_empty());
        assert_eq!(encoder.out_buf.get_mut().capacity(), 0);
        // Repeated flushes without new input must not emit another frame.
        {
            let mut storage = vec![MaybeUninit::uninit(); 9];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            assert!(encoder.flush(&mut output).unwrap());
            assert_eq!(output.written_len(), 0);
        }

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_flush_owned_partial_block_with_short_output() {
        let expected = vec![0; MAX_BLOCK_SIZE / 2];
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let mut input = PartialBuffer::new(expected.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            encoded.extend_from_slice(output.written());
        }

        let required = 8 + snap::raw::max_compress_len(expected.len());

        {
            let mut storage = vec![MaybeUninit::uninit(); required - 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            assert!(encoder.flush(&mut output).unwrap());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert!(encoder.out_buf.get_mut().capacity() > 0);

        {
            let mut storage: [MaybeUninit<u8>; 0] = [];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            assert!(encoder.flush(&mut output).unwrap());
            assert_eq!(output.written_len(), 0);
        }

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_encode_first_frame_one_byte_below_direct_output_boundary() {
        let input = vec![0; MAX_BLOCK_SIZE];
        let mut input_buffer = PartialBuffer::new(input.as_slice());

        let mut storage = vec![MaybeUninit::uninit(); MAX_FRAME_SIZE - 1];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);
        let mut encoder = SnappyEncoder::new();

        encoder.encode(&mut input_buffer, &mut output).unwrap();

        assert!(input_buffer.unwritten().is_empty());
        assert_eq!(output.initialized_len(), output.written_len());
        assert_eq!(encoder.in_buf.get_mut().capacity(), 0);
        assert!(encoder.out_buf.get_mut().capacity() > 0);

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(output.written())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_later_frame_one_byte_below_direct_output_boundary() {
        let mut encoder = SnappyEncoder::new();
        let mut encoded = Vec::new();

        {
            let mut input = PartialBuffer::new(&[][..]);
            let mut storage = vec![MaybeUninit::uninit(); STREAM_FRAME.len() + 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();
            encoded.extend_from_slice(output.written());
        }

        {
            let input = vec![0; MAX_BLOCK_SIZE - 1];
            let mut input = PartialBuffer::new(input.as_slice());
            let mut storage: [MaybeUninit<u8>; 0] = [];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            assert_eq!(encoder.out_buf.get_mut().capacity(), 0);
        }

        {
            let input = [0];
            let mut input = PartialBuffer::new(input.as_slice());
            let mut storage = vec![MaybeUninit::uninit(); MAX_DATA_FRAME_SIZE - 1];
            let mut output = WriteBuffer::new_uninitialized(&mut storage);

            encoder.encode(&mut input, &mut output).unwrap();

            assert!(input.unwritten().is_empty());
            assert_eq!(output.initialized_len(), output.written_len());
            encoded.extend_from_slice(output.written());
        }

        assert!(encoder.out_buf.get_mut().capacity() > 0);

        let mut decoded = Vec::new();
        snap::read::FrameDecoder::new(encoded.as_slice())
            .read_to_end(&mut decoded)
            .unwrap();

        assert_eq!(decoded, vec![0; MAX_BLOCK_SIZE]);
    }
}
