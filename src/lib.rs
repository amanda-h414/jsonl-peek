//! Library half of `jsonl-peek`: a streaming JSONL reader with no
//! third-party dependencies.
//!
//! The binary (`src/main.rs`) is a thin CLI shell over this crate. Modules
//! land here roughly in the order the pipeline needs them - the JSON parser
//! and the line splitter first, since everything else reads through them.

pub mod json;
pub mod lines;
