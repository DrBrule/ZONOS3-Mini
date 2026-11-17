/**
 * Project documents from older studios, brought to the current schema (pure; the MCP server imports it in Node).
 * Schema 2 writes pronunciation groups "[[X-SAMPA]]" where schema 1 wrote "(X-SAMPA)".
 */
import { xsampaToIpa } from '../lib/pron';
import type { ProjectDoc } from './projectSchema';

/** Schema 1's groups: a "(" with a later ")" and no "(" between them, whose body spells IPA and has no square bracket. */
function oldGroups(text: string): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== '(') {
      i++;
      continue;
    }
    const close = text.indexOf(')', i + 1);
    const nested = text.indexOf('(', i + 1);
    if (close < 0 || (nested >= 0 && nested < close)) {
      i++;
      continue;
    }
    const body = text.slice(i + 1, close);
    if (!/[[\]]/.test(body) && xsampaToIpa(body) !== '') out.push({ start: i, end: close + 1 });
    i = close + 1;
  }
  return out;
}

/** `text` with its schema-1 groups in brackets, and how an offset into it moves (each group grows by one at each end). */
function toBrackets(text: string): { text: string; shift: (p: number) => number } {
  const groups = oldGroups(text);
  let out = '';
  let pos = 0;
  for (const g of groups) {
    out += `${text.slice(pos, g.start)}[[${text.slice(g.start + 1, g.end - 1)}]]`;
    pos = g.end;
  }
  out += text.slice(pos);
  return { text: out, shift: (p) => p + groups.reduce((n, g) => n + Number(p > g.start) + Number(p >= g.end), 0) };
}

const converted = (text: string) => toBrackets(text).text;

/** The document at schema 2 (a copy; unchanged when it already is). Stored requests keep their sent form. */
export function migrateDoc(doc: ProjectDoc): ProjectDoc {
  if (doc.schema >= 2) return doc;
  const d = structuredClone(doc);
  d.schema = 2;
  for (const line of d.direct.script.lines) {
    const { text, shift } = toBrackets(line.text);
    line.text = text;
    for (const s of line.spans) {
      s.start = shift(s.start);
      s.end = shift(s.end);
    }
  }
  for (const list of Object.values(d.direct.takes)) for (const t of list) t.text = converted(t.text);
  d.cast.previewText = converted(d.cast.previewText);
  for (const p of d.cast.personas) p.signature = converted(p.signature);
  for (const v of d.edit.doc?.versions ?? []) if (v.display !== undefined) v.display = converted(v.display);
  for (const l of d.scene?.scene.lines ?? []) l.text = converted(l.text);
  return d;
}
