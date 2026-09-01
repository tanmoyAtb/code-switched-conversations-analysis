import type { Label } from './types.js';

/** Bengali block, U+0980–U+09FF. */
const BENGALI = /[ঀ-৿]/;
const LATIN = /[A-Za-z]/;

export type Script = 'bengali' | 'latin' | 'both' | 'neither';

export function scriptOf(text: string): Script {
  const bengali = BENGALI.test(text);
  const latin = LATIN.test(text);
  if (bengali && latin) return 'both';
  if (bengali) return 'bengali';
  if (latin) return 'latin';
  return 'neither';
}

/**
 * Which labels an annotator may choose, given the alphabet.
 *
 * Script is machine-readable; language is not. That asymmetry is the whole project, and it
 * is also what keeps this constraint honest: two of the three labels name a script in their
 * own definition, so the alphabet rules them in or out without touching the language
 * question. `romanized_bangla` is Bangla in LATIN script, so it cannot apply to Bengali
 * characters. `bangla` is Bangla in BENGALI script, so it cannot apply to Latin ones.
 * `english` survives everywhere, because English does get transliterated into Bengali script.
 *
 * What is left for Latin-only text — english vs romanized_bangla — is the one genuine
 * judgement in the scheme.
 */
export function optionsFor(script: Script): readonly Label[] {
  switch (script) {
    case 'latin':
      return ['english', 'romanized_bangla'];
    case 'bengali':
      return ['bangla', 'english'];
    // Both alphabets in one message, or neither: rare, and not settled by script. Offer
    // everything rather than guess.
    case 'both':
    case 'neither':
      return ['english', 'bangla', 'romanized_bangla'];
  }
}

/**
 * The label the alphabet settles on its own, or null where a human must decide.
 *
 * Bengali-script messages are Bangla in all but a handful of transliterated cases, so
 * spending two annotators' attention on every one of them buys almost nothing. They render
 * as context with the assumed label shown, and can still be opened and changed — the
 * assumption is a default, not a verdict. `both` is not treated this way: mixed alphabets
 * are exactly where the assumption would be least safe.
 */
export function autoLabel(script: Script): Label | null {
  return script === 'bengali' ? 'bangla' : null;
}

/** True where a human click is required for the thread to count as finished. */
export function needsJudgement(script: Script): boolean {
  return autoLabel(script) === null;
}
