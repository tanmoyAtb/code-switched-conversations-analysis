import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { config } from './config.js';
import type { AnnotationEntry, AnnotationFile, AnnotatorId } from './types.js';

const ANNOTATOR_DIR = /^annotator_(\d+)$/;
const THREAD_FILE = /^thread_(\d+)\.json$/;

function annotatorDir(annotatorId: AnnotatorId): string {
  return path.join(config.annotationsDir, `annotator_${annotatorId}`);
}

function threadPath(annotatorId: AnnotatorId, threadId: number): string {
  return path.join(annotatorDir(annotatorId), `thread_${String(threadId).padStart(4, '0')}.json`);
}

function key(annotatorId: AnnotatorId, threadId: number): string {
  return `${annotatorId}/${threadId}`;
}

/**
 * The annotation files on disk, mirrored in memory.
 *
 * Every click is its own write. Nothing is buffered in the page and Next saves nothing, so
 * closing the laptop mid-conversation costs at most the message in progress.
 */
export class AnnotationStore {
  private readonly files = new Map<string, AnnotationFile>();
  /** Per-file write chain. Two rapid clicks on one thread must land in order. */
  private readonly writes = new Map<string, Promise<void>>();

  static async load(): Promise<AnnotationStore> {
    const store = new AnnotationStore();
    await mkdir(config.annotationsDir, { recursive: true });

    for (const dirent of await readdir(config.annotationsDir, { withFileTypes: true })) {
      const dirMatch = ANNOTATOR_DIR.exec(dirent.name);
      if (!dirent.isDirectory() || dirMatch === null) continue;
      const dir = path.join(config.annotationsDir, dirent.name);

      for (const name of await readdir(dir)) {
        if (!THREAD_FILE.test(name)) continue;
        const parsed = JSON.parse(await readFile(path.join(dir, name), 'utf8')) as AnnotationFile;
        store.files.set(key(parsed.annotator_id, parsed.thread_id), parsed);
      }
    }
    return store;
  }

  get(annotatorId: AnnotatorId, threadId: number): AnnotationFile | undefined {
    return this.files.get(key(annotatorId, threadId));
  }

  /** Who has opened this thread. Derived, so it cannot disagree with what is on disk. */
  annotatorsFor(threadId: number): AnnotatorId[] {
    const ids: AnnotatorId[] = [];
    for (const file of this.files.values()) {
      if (file.thread_id === threadId) ids.push(file.annotator_id);
    }
    return ids.sort((a, b) => a - b);
  }

  entryCount(annotatorId: AnnotatorId): number {
    let total = 0;
    for (const file of this.files.values()) {
      if (file.annotator_id === annotatorId) total += file.annotations.length;
    }
    return total;
  }

  totalEntries(): number {
    let total = 0;
    for (const file of this.files.values()) total += file.annotations.length;
    return total;
  }

  /**
   * Record one decision, replacing any earlier one for the same message.
   *
   * Changing your mind is legitimate — you read turn 5 and realise turn 4 was English after
   * all — so this overwrites in place. `ms_spent` is kept from the first decision rather than
   * re-timed by the correction, because the interesting quantity is how long the judgement
   * took, not how long the fix did.
   */
  async record(params: {
    annotator_id: AnnotatorId;
    thread_id: number;
    business_id: number;
    is_control: boolean;
    entry: AnnotationEntry;
  }): Promise<AnnotationFile> {
    const k = key(params.annotator_id, params.thread_id);
    const file: AnnotationFile = this.files.get(k) ?? {
      annotator_id: params.annotator_id,
      thread_id: params.thread_id,
      business_id: params.business_id,
      is_control: params.is_control,
      updated_at: '',
      annotations: [],
    };

    // Replace the entry rather than mutating it, so what is written always has exactly the
    // current shape. Mutating fields lets anything left in an older file on disk — a retired
    // status, a dropped flag — survive every subsequent write and outlive the code that
    // understood it.
    const index = file.annotations.findIndex((a) => a.message_id === params.entry.message_id);
    if (index === -1) file.annotations.push(params.entry);
    else file.annotations[index] = params.entry;

    file.annotations.sort((a, b) => a.message_id - b.message_id);
    file.updated_at = new Date().toISOString();
    this.files.set(k, file);

    await this.persist(k, params.annotator_id, params.thread_id, file);
    return file;
  }

  /**
   * Write via a temp file and rename, so a crash mid-write cannot leave a half-written
   * conversation behind: the reader either sees the previous version or the new one.
   */
  private async persist(
    k: string,
    annotatorId: AnnotatorId,
    threadId: number,
    file: AnnotationFile,
  ): Promise<void> {
    const chained = (this.writes.get(k) ?? Promise.resolve()).then(async () => {
      const target = threadPath(annotatorId, threadId);
      const temp = `${target}.tmp-${process.pid}`;
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(temp, `${JSON.stringify(file, null, 2)}\n`, 'utf8');
      await rename(temp, target);
    });

    this.writes.set(k, chained.catch(() => undefined));
    await chained;
  }
}
