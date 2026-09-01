import { createApp } from './app.js';
import { config } from './config.js';
import { loadCorpus } from './corpus.js';
import { AnnotationStore } from './store.js';

async function main(): Promise<void> {
  const [corpus, store] = await Promise.all([loadCorpus(), AnnotationStore.load()]);
  console.log(
    `corpus: ${corpus.message_count} messages in ${corpus.thread_ids.length} conversations; ` +
      `${store.totalEntries()} annotations on disk`,
  );

  const server = createApp(corpus, store).listen(config.port, () => {
    console.log(`annotator listening on http://localhost:${config.port}`);
  });

  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.on(signal, () => {
      server.close(() => process.exit(0));
    });
  }
}

main().catch((err: unknown) => {
  console.error('Failed to start:', err);
  process.exit(1);
});
