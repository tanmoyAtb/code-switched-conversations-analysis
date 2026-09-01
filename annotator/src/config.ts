import path from 'node:path';
import 'dotenv/config';

function env(name: string, fallback: string): string {
  const value = process.env[name];
  return value === undefined || value === '' ? fallback : value;
}

/** annotator/ — one level up from src/ or dist/. */
export const appRoot = path.join(import.meta.dirname, '..');
const repoRoot = path.join(appRoot, '..');

export const config = {
  port: Number(env('PORT', '3000')),
  corpusPath: path.resolve(repoRoot, env('CORPUS_PATH', 'data/texts_no_annotation.jsonl')),
  annotationsDir: path.resolve(repoRoot, env('ANNOTATIONS_DIR', 'data/annotations')),
} as const;
