// node --test: the web tokenizer must give exactly the ids the Rust `tokenizers` crate gives.
// Needs tokenizer.json from Xenova/musicgen-small; TOKENIZER_JSON points at it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { Tokenizer } from "./tokenizer.js";

const path =
  process.env.TOKENIZER_JSON ?? `${homedir()}/Library/Application Support/Driftscape/models/musicgen-small/tokenizer.json`;
const fixtures = JSON.parse(readFileSync(new URL("./tokenizer.fixtures.json", import.meta.url), "utf8"));

test("matches the Rust tokenizer", { skip: !existsSync(path) && `no tokenizer.json at ${path}` }, () => {
  const tokenizer = new Tokenizer(JSON.parse(readFileSync(path, "utf8")));
  for (const { prompt, ids } of fixtures) {
    assert.deepEqual(tokenizer.encode(prompt), ids, JSON.stringify(prompt));
  }
});
