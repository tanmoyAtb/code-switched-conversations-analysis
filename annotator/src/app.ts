import path from 'node:path';
import express, { type Express, type NextFunction, type Request, type Response } from 'express';
import cookieParser from 'cookie-parser';
import { appRoot } from './config.js';
import type { Corpus } from './corpus.js';
import { isComplete, nextThreadFor, progressFor } from './queue.js';
import { autoLabel, optionsFor } from './script.js';
import type { AnnotationStore } from './store.js';
import {
  LABEL_DEFINITIONS,
  LABEL_SHORT,
  isLabel,
  parseAnnotatorId,
  type AnnotationEntry,
  type AnnotatorId,
} from './types.js';

const COOKIE = 'annotator_id';

/**
 * The annotator id is re-asked each session rather than remembered indefinitely: it costs one
 * keystroke and it stops annotator 2 silently inheriting annotator 1's laptop session.
 */
const COOKIE_OPTIONS = { httpOnly: true, sameSite: 'lax' } as const;

function annotatorFrom(req: Request): AnnotatorId | null {
  return parseAnnotatorId((req.cookies as Record<string, unknown>)[COOKIE]);
}

export function createApp(corpus: Corpus, store: AnnotationStore): Express {
  const app = express();

  app.set('view engine', 'ejs');
  app.set('views', path.join(appRoot, 'views'));

  app.use(express.urlencoded({ extended: false }));
  app.use(express.json());
  app.use(cookieParser());
  app.use(express.static(path.join(appRoot, 'public')));

  app.get('/', (req, res) => {
    if (annotatorFrom(req) !== null) return res.redirect('/annotate');
    res.render('login', { labels: LABEL_DEFINITIONS, error: null });
  });

  app.post('/session', (req, res) => {
    const annotatorId = parseAnnotatorId((req.body as Record<string, unknown>)['annotator_id']);
    if (annotatorId === null) {
      return res
        .status(400)
        .render('login', { labels: LABEL_DEFINITIONS, error: 'Enter a whole number from 1 to 99.' });
    }
    res.cookie(COOKIE, String(annotatorId), COOKIE_OPTIONS);
    res.redirect('/annotate');
  });

  app.post('/logout', (_req, res) => {
    res.clearCookie(COOKIE);
    res.redirect('/');
  });

  app.get('/annotate', (req, res) => {
    const annotatorId = annotatorFrom(req);
    if (annotatorId === null) return res.redirect('/');

    const thread = nextThreadFor(corpus, store, annotatorId);
    if (thread === null) {
      return res.render('done', {
        annotatorId,
        progress: progressFor(corpus, store, annotatorId),
      });
    }
    res.redirect(`/annotate/${thread.thread_id}`);
  });

  app.get('/annotate/:threadId', (req, res) => {
    const annotatorId = annotatorFrom(req);
    if (annotatorId === null) return res.redirect('/');

    const thread = corpus.threads.get(Number(req.params.threadId));
    if (thread === undefined) return res.status(404).send('No such conversation');

    const file = store.get(annotatorId, thread.thread_id);
    const entries = new Map(file?.annotations.map((a) => [a.message_id, a]) ?? []);

    res.render('annotate', {
      annotatorId,
      thread,
      rows: thread.messages.map((message) => ({
        message,
        options: optionsFor(message.script),
        // The label the alphabet assumes when nobody clicks. Shown so the annotator can
        // disagree with it, which is the only way a wrong assumption ever gets corrected.
        assumed: autoLabel(message.script),
        entry: entries.get(message.message_id) ?? null,
      })),
      remaining: thread.annotatable_ids.filter((id) => !entries.has(id)).length,
      complete: isComplete(thread, file),
      progress: progressFor(corpus, store, annotatorId),
      shortLabels: LABEL_SHORT,
      definitions: LABEL_DEFINITIONS,
    });
  });

  /**
   * One decision, one write. The response carries the recomputed remaining count so the page
   * never has to guess whether the conversation is finished.
   */
  app.post('/api/annotate', async (req, res) => {
    const annotatorId = annotatorFrom(req);
    if (annotatorId === null) return res.status(401).json({ error: 'No annotator id' });

    const body = req.body as Record<string, unknown>;
    const thread = corpus.threads.get(Number(body['thread_id']));
    const message = thread?.messages.find((m) => m.message_id === Number(body['message_id']));
    if (thread === undefined || message === undefined) {
      return res.status(400).json({ error: 'Unknown message' });
    }

    const label = body['label'];
    if (!isLabel(label) || !optionsFor(message.script).includes(label)) {
      return res.status(400).json({ error: 'Label not available for this script' });
    }

    const previous = store.get(annotatorId, thread.thread_id)?.annotations
      .find((a) => a.message_id === message.message_id);

    const entry: AnnotationEntry = {
      message_id: message.message_id,
      turn: message.turn,
      label,
      // Kept from the first decision: the interesting quantity is how long the judgement
      // took, not how long a later correction did.
      ms_spent: previous?.ms_spent ?? Math.max(0, Math.trunc(Number(body['ms_spent']) || 0)),
    };

    const file = await store.record({
      annotator_id: annotatorId,
      thread_id: thread.thread_id,
      business_id: thread.business_id,
      is_control: thread.is_control,
      entry,
    });

    const answered = new Set(file.annotations.map((a) => a.message_id));
    res.json({
      ok: true,
      entry,
      remaining: thread.annotatable_ids.filter((id) => !answered.has(id)).length,
    });
  });

  app.get('/health', (_req, res) => {
    res.json({
      status: 'ok',
      threads: corpus.thread_ids.length,
      messages: corpus.message_count,
      annotations: store.totalEntries(),
    });
  });

  app.use((_req, res) => {
    res.status(404).send('Not found');
  });

  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    console.error(err);
    res.status(500).send('Internal error');
  });

  return app;
}
