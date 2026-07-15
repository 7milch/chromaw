import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { FullConfig } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Where the running server's baseURL/token are published for spec files to
 * read (e2e/fixtures.ts). A plain JSON file rather than process.env,
 * because Playwright evaluates playwright.config.ts (which captures
 * `use.baseURL` as a plain value) before globalSetup runs, so env vars set
 * here would arrive too late for the config object; a file read lazily by
 * fixtures.ts at test time does not have that ordering problem. */
export const SERVER_INFO_PATH = path.resolve(__dirname, ".server-info.json");

/**
 * Global setup for the Playwright suite (M4-5, technical-spec §12): seeds a
 * throwaway ChromaDB directory with a small "e2e" collection, then launches
 * the real `chromaw --write` CLI as a subprocess against it -- the same
 * pattern used by tests/test_e2e_server.py on the Python side, just driven
 * from the frontend test runner. baseURL/env for the tests themselves is
 * written out via `process.env` so individual spec files (and
 * global-teardown) can find the running server and token without importing
 * this file.
 */

const SEED_SCRIPT = `
import chromadb
import sys

path = sys.argv[1]
client = chromadb.PersistentClient(path=path)

# Main seed collection: read/search/edit specs rely on this having exactly
# these 5 untouched records, so destructive tests must not touch it.
collection = client.create_collection("e2e", metadata={"purpose": "playwright e2e"})
ids = [f"rec-{i}" for i in range(5)]
documents = [f"e2e document number {i}" for i in range(5)]
metadatas = [{"idx": i, "category": "even" if i % 2 == 0 else "odd"} for i in range(5)]
embeddings = [[float(i), float(i) + 0.5, float(i) + 1.0] for i in range(5)]
collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
print("seeded", collection.count(), "records")

# Scratch collection for the record/collection-deletion guard tests
# (delete.spec.ts), kept separate so those destructive operations can't
# affect the specs above regardless of file execution order.
delete_collection = client.create_collection("e2e-delete")
delete_ids = [f"del-{i}" for i in range(3)]
delete_collection.add(
    ids=delete_ids,
    documents=[f"scratch document {i}" for i in range(3)],
    metadatas=[{"idx": i} for i in range(3)],
    embeddings=[[float(i)] * 3 for i in range(3)],
)
print("seeded", delete_collection.count(), "scratch records")
`;

function findRepoRoot(): string {
  // web/e2e/global-setup.ts -> repo root is two levels up.
  return path.resolve(__dirname, "..", "..");
}

async function waitForServer(baseURL: string, timeoutMs = 20000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${baseURL}/`);
      if (res.status === 200) return;
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`chromaw server did not become ready in time: ${lastErr}`);
}

async function fetchToken(baseURL: string): Promise<string> {
  const res = await fetch(`${baseURL}/`);
  const html = await res.text();
  const match = html.match(/<meta name="chromaw-token" content="([^"]+)">/);
  if (!match) {
    throw new Error(`no chromaw-token meta tag found in index.html:\n${html.slice(0, 500)}`);
  }
  return match[1];
}

export default async function globalSetup(_config: FullConfig): Promise<() => Promise<void>> {
  const repoRoot = findRepoRoot();
  const tmpDir = mkdtempSync(path.join(tmpdir(), "chromaw-e2e-"));

  // Seed the ChromaDB directory with a Python one-liner via `uv run`, mirroring
  // the "uv run python -c" seeding approach called for by the task.
  execFileSync("uv", ["run", "python", "-c", SEED_SCRIPT, tmpDir], {
    cwd: repoRoot,
    stdio: "inherit",
  });

  const proc = spawn(
    "uv",
    ["run", "chromaw", tmpDir, "--write", "--no-open", "--port", "0"],
    { cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] }
  );

  let stdoutBuf = "";
  const port = await new Promise<number>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error(`chromaw did not report a running URL in time. output so far:\n${stdoutBuf}`));
    }, 20000);

    proc.stdout.on("data", (chunk: Buffer) => {
      stdoutBuf += chunk.toString();
      const match = stdoutBuf.match(/chromaw is running at http:\/\/[^:]+:(\d+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(Number(match[1]));
      }
    });
    proc.stderr.on("data", (chunk: Buffer) => {
      stdoutBuf += chunk.toString();
    });
    proc.on("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`chromaw process exited early (code ${code}). output so far:\n${stdoutBuf}`));
    });
  });

  const baseURL = `http://127.0.0.1:${port}`;
  await waitForServer(baseURL);
  const token = await fetchToken(baseURL);

  // Published for spec files (running in separate worker processes) via
  // e2e/fixtures.ts, which reads this file lazily at test time. See
  // SERVER_INFO_PATH's docstring for why this isn't done via process.env.
  writeFileSync(SERVER_INFO_PATH, JSON.stringify({ baseURL, token }), "utf-8");

  // Returned function is invoked once as global teardown, in this same
  // process, so it can close over proc/tmpDir directly.
  return async () => {
    try {
      proc.kill("SIGTERM");
    } catch {
      // already exited
    }
    rmSync(tmpDir, { recursive: true, force: true });
    rmSync(SERVER_INFO_PATH, { force: true });
  };
}
