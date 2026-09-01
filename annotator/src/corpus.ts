import { readFile } from "node:fs/promises";
import { config } from "./config.js";
import { needsJudgement, scriptOf, type Script } from "./script.js";
import type { Message } from "./types.js";

export interface CorpusMessage extends Message {
  script: Script;
  gap_before: number;
  annotatable: boolean;
}

export interface Thread {
  thread_id: number;
  business_id: number;
  is_control: boolean;
  messages: CorpusMessage[];
  /** message_ids that must each carry an entry before the thread counts as done. */
  annotatable_ids: number[];
}

export interface Corpus {
  threads: Map<number, Thread>;
  /** Ascending thread_id. Queue order is derived from this, so it must be stable. */
  thread_ids: number[];
  message_count: number;
}

const CONTROL_EVERY = 30;

export function isControl(threadId: number): boolean {
  return threadId % CONTROL_EVERY === 0;
}

export async function loadCorpus(): Promise<Corpus> {
  const raw = await readFile(config.corpusPath, "utf8");
  const rows: Message[] = raw
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as Message);

  const byThread = new Map<number, Message[]>();
  for (const row of rows) {
    const bucket = byThread.get(row.thread_id);
    if (bucket) bucket.push(row);
    else byThread.set(row.thread_id, [row]);
  }

  const threads = new Map<number, Thread>();
  for (const [threadId, group] of byThread) {
    group.sort((a, b) => a.turn - b.turn);

    const messages: CorpusMessage[] = group.map((row, index) => {
      const previous = index === 0 ? undefined : group[index - 1];
      const script = scriptOf(row.text);
      return {
        ...row,
        script,
        gap_before:
          previous === undefined
            ? 0
            : Math.max(0, row.turn - previous.turn - 1),
        annotatable: needsJudgement(script),
      };
    });

    const first = messages[0];
    if (first === undefined) continue;

    threads.set(threadId, {
      thread_id: threadId,
      business_id: first.business_id,
      is_control: isControl(threadId),
      messages,
      annotatable_ids: messages
        .filter((m) => m.annotatable)
        .map((m) => m.message_id),
    });
  }

  return {
    threads,
    thread_ids: [...threads.keys()].sort((a, b) => a - b),
    message_count: rows.length,
  };
}
