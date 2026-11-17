import { describe, expect, it } from 'vitest';
import { homePin } from './atlas';
import { migrateDoc } from './migrate';
import type { ProjectDoc, StoredEditDoc } from './projectSchema';

const OLD = 'I met (S@vQn) and (k@Uv).';
const NEW = 'I met [[S@vQn]] and [[k@Uv]].';

function schema1(text: string): ProjectDoc {
  const and = text.indexOf('and');
  const design = { baseId: 'spk', mode: 'embedding' as const, values: {}, pin: homePin() };
  return {
    schema: 1,
    id: 'p1',
    name: 'Test',
    createdAt: 0,
    updatedAt: 0,
    settings: { language: 'en_us', voiceMode: 'embedding', decoder: null, laci: 'off' },
    voices: [],
    cast: { design, personas: [{ id: 'c1', name: 'C', design, signature: text, createdAt: 0 }], origins: {}, previewText: text },
    direct: {
      script: { lines: [{ id: 'l1', text, spans: [{ id: 's1', start: and, end: and + 3, values: { happy: 2 } }] }], baseline: {}, voice: null, gap: 0.35 },
      takes: { l1: [{ id: 't1', text, seed: 1, request: { text: '\ue000ʃəvɒn\ue001' }, audio: { ttfbMs: 1, totalMs: 2, media: 'a'.repeat(64) }, createdAt: 0, starred: true }] },
    },
    edit: { doc: { versions: [{ display: text }] } as unknown as StoredEditDoc, clipLanguages: {} },
    media: {},
  };
}

describe('migrateDoc', () => {
  it('writes schema 1 groups as [[…]], spans moving with their words', () => {
    const d = migrateDoc(schema1(OLD));
    expect(d.schema).toBe(2);
    const line = d.direct.script.lines[0];
    expect(line.text).toBe(NEW);
    expect(line.text.slice(line.spans[0].start, line.spans[0].end)).toBe('and');
    expect(line.spans[0].start).toBe(OLD.indexOf('and') + 2);
    expect(d.direct.takes.l1[0].text).toBe(NEW);
    expect(d.direct.takes.l1[0].request.text).toBe('\ue000ʃəvɒn\ue001');
    expect(d.cast.previewText).toBe(NEW);
    expect(d.cast.personas[0].signature).toBe(NEW);
    expect(d.edit.doc!.versions[0].display).toBe(NEW);
    expect(migrateDoc(d)).toBe(d);
  });

  it('converts what the old parser took as a group, and nothing else', () => {
    const text = (t: string) => migrateDoc(schema1(t)).direct.script.lines[0].text;
    expect(text('He (laughs!) on')).toBe('He [[laughs!]] on');
    expect(text('an open ( paren')).toBe('an open ( paren');
    expect(text('empty () here')).toBe('empty () here');
    expect(text('(see [1])')).toBe('(see [1])');
  });
});
