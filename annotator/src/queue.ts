import type { Corpus, Thread } from './corpus.js';
import type { AnnotationStore } from './store.js';
import type { AnnotationFile, AnnotatorId } from './types.js';

/** Two annotators per conversation, so every judgement has something to agree with. */
const ANNOTATORS_PER_THREAD = 2;

export function isComplete(thread: Thread, file: AnnotationFile | undefined): boolean {
  if (file === undefined) return thread.annotatable_ids.length === 0;
  const answered = new Set(file.annotations.map((a) => a.message_id));
  return thread.annotatable_ids.every((id) => answered.has(id));
}

/**
 * The next conversation for this annotator, or null when there is nothing left for them.
 *
 * Order of preference:
 *   1. A conversation they already started and left unfinished. Resuming beats opening.
 *   2. A conversation exactly one other annotator has done — finishing a pair yields an
 *      agreement measurement, whereas opening a fresh conversation yields a lone opinion.
 *   3. Anything untouched.
 *
 * Controls ignore the cap: they are the conversations every annotator sees, which is what
 * makes agreement comparable between two people who otherwise share no work.
 */
export function nextThreadFor(
  corpus: Corpus,
  store: AnnotationStore,
  annotatorId: AnnotatorId,
): Thread | null {
  let pairable: Thread | null = null;
  let fresh: Thread | null = null;

  for (const threadId of corpus.thread_ids) {
    const thread = corpus.threads.get(threadId);
    if (thread === undefined || thread.annotatable_ids.length === 0) continue;

    const own = store.get(annotatorId, threadId);
    if (own !== undefined) {
      if (!isComplete(thread, own)) return thread;
      continue;
    }

    const others = store.annotatorsFor(threadId).filter((id) => id !== annotatorId).length;
    if (!thread.is_control && others >= ANNOTATORS_PER_THREAD) continue;

    if (others === 1) pairable ??= thread;
    else fresh ??= thread;
  }

  return pairable ?? fresh;
}

export interface Progress {
  threads_done: number;
  messages_done: number;
  threads_total: number;
}

export function progressFor(
  corpus: Corpus,
  store: AnnotationStore,
  annotatorId: AnnotatorId,
): Progress {
  let done = 0;
  for (const thread of corpus.threads.values()) {
    if (thread.annotatable_ids.length === 0) continue;
    if (isComplete(thread, store.get(annotatorId, thread.thread_id))) done += 1;
  }
  return {
    threads_done: done,
    messages_done: store.entryCount(annotatorId),
    threads_total: corpus.thread_ids.length,
  };
}
