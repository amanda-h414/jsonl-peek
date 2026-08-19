//! A reusable-buffer line splitter for JSONL input.
//!
//! `BufRead::lines()` allocates a fresh `String` for every line, which is
//! wasteful when scanning a multi-gigabyte file one record at a time.
//! [`LineReader`] reuses a single internal buffer across calls instead, and
//! deals with the three things real JSONL files do that a naive split on
//! `\n` gets wrong: a UTF-8 BOM on the first line, CRLF line endings, and a
//! final line with no trailing newline at all.
//!
//! It works with raw bytes, not `&str` - UTF-8 validation is the caller's
//! job, since a caller reporting a broken line usually wants to say exactly
//! where the invalid byte is, not just that the line failed to decode.
//!
//! ```
//! use jsonl_peek::lines::LineReader;
//!
//! let mut reader = LineReader::new(b"\xEF\xBB\xBFa\r\nb\n".as_slice());
//! assert_eq!(reader.next_line().unwrap(), Some(&b"a"[..]));
//! assert_eq!(reader.next_line().unwrap(), Some(&b"b"[..]));
//! assert_eq!(reader.next_line().unwrap(), None);
//! ```

use std::io::{self, BufRead};

const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Splits a byte stream into JSONL records, reusing one buffer per line.
///
/// Each call to [`next_line`](LineReader::next_line) overwrites the
/// previously returned slice, so the result must be consumed (or copied)
/// before the next call.
pub struct LineReader<R> {
    inner: R,
    buf: Vec<u8>,
    lines_read: usize,
    stripped_bom: bool,
}

impl<R: BufRead> LineReader<R> {
    /// Wraps a buffered reader.
    pub fn new(inner: R) -> Self {
        LineReader {
            inner,
            buf: Vec::new(),
            lines_read: 0,
            stripped_bom: false,
        }
    }

    /// Reads the next line, with its terminator stripped.
    ///
    /// Recognises `\n` and `\r\n`; a final line with neither is still
    /// returned. A UTF-8 BOM at the very start of the stream is stripped
    /// from the first line, wherever the first line ends. Returns `Ok(None)`
    /// once the stream is exhausted; a stream that ends immediately after a
    /// `\n` does not produce a trailing empty line.
    pub fn next_line(&mut self) -> io::Result<Option<&[u8]>> {
        self.buf.clear();
        let n = self.inner.read_until(b'\n', &mut self.buf)?;
        if n == 0 {
            return Ok(None);
        }
        if self.buf.last() == Some(&b'\n') {
            self.buf.pop();
            if self.buf.last() == Some(&b'\r') {
                self.buf.pop();
            }
        }
        if !self.stripped_bom {
            self.stripped_bom = true;
            if self.buf.starts_with(&BOM) {
                self.buf.drain(0..BOM.len());
            }
        }
        self.lines_read += 1;
        Ok(Some(&self.buf))
    }

    /// The number of lines returned so far by [`next_line`](LineReader::next_line).
    pub fn line_number(&self) -> usize {
        self.lines_read
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &[u8]) -> Vec<Vec<u8>> {
        let mut reader = LineReader::new(input);
        let mut out = Vec::new();
        while let Some(line) = reader.next_line().unwrap() {
            out.push(line.to_vec());
        }
        out
    }

    #[test]
    fn splits_on_newline() {
        assert_eq!(collect(b"a\nb\nc\n"), vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
    }

    #[test]
    fn keeps_final_line_without_trailing_newline() {
        assert_eq!(collect(b"a\nb"), vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn trailing_newline_does_not_add_an_empty_line() {
        assert_eq!(collect(b"a\n"), vec![b"a".to_vec()]);
    }

    #[test]
    fn empty_input_yields_no_lines() {
        assert_eq!(collect(b""), Vec::<Vec<u8>>::new());
    }

    #[test]
    fn blank_lines_are_preserved() {
        assert_eq!(
            collect(b"a\n\nb\n"),
            vec![b"a".to_vec(), Vec::new(), b"b".to_vec()]
        );
    }

    #[test]
    fn strips_crlf() {
        assert_eq!(collect(b"a\r\nb\r\n"), vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn bare_cr_is_left_alone() {
        // Only CRLF is a line ending here; a lone CR is data.
        assert_eq!(collect(b"a\rb\n"), vec![b"a\rb".to_vec()]);
    }

    #[test]
    fn strips_bom_from_first_line_only() {
        let mut reader = LineReader::new(&b"\xEF\xBB\xBFa\nb\xEF\xBB\xBF\n"[..]);
        assert_eq!(reader.next_line().unwrap(), Some(&b"a"[..]));
        assert_eq!(reader.next_line().unwrap(), Some(&b"b\xEF\xBB\xBF"[..]));
        assert_eq!(reader.next_line().unwrap(), None);
    }

    #[test]
    fn bom_survives_a_file_with_a_single_unterminated_line() {
        let mut reader = LineReader::new(&b"\xEF\xBB\xBFonly"[..]);
        assert_eq!(reader.next_line().unwrap(), Some(&b"only"[..]));
    }

    #[test]
    fn tracks_line_number() {
        let mut reader = LineReader::new(&b"a\nb\nc"[..]);
        assert_eq!(reader.line_number(), 0);
        reader.next_line().unwrap();
        assert_eq!(reader.line_number(), 1);
        reader.next_line().unwrap();
        reader.next_line().unwrap();
        assert_eq!(reader.line_number(), 3);
        reader.next_line().unwrap();
        assert_eq!(reader.line_number(), 3);
    }

    #[test]
    fn buffer_is_reused_across_calls() {
        // Not observable from the API directly, but the capacity should stick
        // around rather than being reallocated every call.
        let mut reader = LineReader::new(&b"aaaaaaaaaa\nb\n"[..]);
        reader.next_line().unwrap();
        let cap = reader.buf.capacity();
        reader.next_line().unwrap();
        assert!(reader.buf.capacity() <= cap);
    }
}
