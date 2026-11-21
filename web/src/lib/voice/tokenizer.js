// The T5 tokenizer MusicGen uses, as specified by its tokenizer.json: SentencePiece Unigram.
//
//   1. normalize (NFKC; the file's precompiled charsmap is NFKC plus details that don't arise in prompts),
//   2. split on whitespace, and prefix every word with "▁" (Metaspace, prepend_scheme "always"),
//   3. segment each word into vocabulary pieces maximising the summed log-probabilities (Viterbi); a character that
//      no piece covers becomes <unk>, and runs of them are fused into one,
//   4. append </s>.
//
// tokenizer.test.js checks it against ids from the Rust `tokenizers` crate (tokenizer.fixtures.json).

const SPACE = "▁";
const UNK_PENALTY = 10;

export class Tokenizer {
  /** @param {any} json the parsed tokenizer.json */
  constructor(json) {
    const model = json.model;
    if (model.type !== "Unigram") throw new Error(`expected a Unigram tokenizer, got ${model.type}`);
    /** @type {Map<string, [number, number]>} piece → [id, score] */
    this.pieces = new Map();
    let minScore = Infinity;
    let maxLen = 0;
    model.vocab.forEach(([piece, score], id) => {
      this.pieces.set(piece, [id, score]);
      if (score < minScore) minScore = score;
      maxLen = Math.max(maxLen, [...piece].length);
    });
    this.unkId = model.unk_id;
    this.unkScore = minScore - UNK_PENALTY;
    this.maxLen = maxLen;
    this.eosId = this.pieces.get("</s>")[0];
  }

  /** @param {string} text @returns {number[]} */
  encode(text) {
    const ids = [];
    for (const word of text.normalize("NFKC").split(/\s+/).filter(Boolean)) {
      ids.push(...this.#segment(SPACE + word));
    }
    ids.push(this.eosId);
    return ids;
  }

  /** Viterbi over one pre-token. */
  #segment(word) {
    const chars = [...word];
    const n = chars.length;
    // best[i]: best score for chars[0..i]; back[i]: [start, id] of the last piece.
    const best = new Float64Array(n + 1).fill(-Infinity);
    const back = new Array(n + 1);
    best[0] = 0;
    for (let end = 1; end <= n; end++) {
      for (let start = Math.max(0, end - this.maxLen); start < end; start++) {
        if (best[start] === -Infinity) continue;
        const hit = this.pieces.get(chars.slice(start, end).join(""));
        if (hit && best[start] + hit[1] > best[end]) {
          best[end] = best[start] + hit[1];
          back[end] = [start, hit[0]];
        }
      }
      // A single character no piece covers is unknown.
      if (best[end] === -Infinity && best[end - 1] !== -Infinity) {
        best[end] = best[end - 1] + this.unkScore;
        back[end] = [end - 1, this.unkId];
      }
    }
    const ids = [];
    for (let end = n; end > 0; ) {
      const [start, id] = back[end];
      // Fuse consecutive unknowns, as SentencePiece does.
      if (!(id === this.unkId && ids.length && ids[ids.length - 1] === this.unkId)) ids.push(id);
      end = start;
    }
    return ids.reverse();
  }
}
