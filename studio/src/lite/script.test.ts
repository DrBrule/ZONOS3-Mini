import { describe, expect, it } from 'vitest';
import type { PronTokens } from '../lib/pron';
import phonemes from '../fixtures/phonemes.json';
import nightFerry from './night-ferry.txt?raw';
import type { ControlsSpec } from '../api';
import fixture from '../fixtures/controls.json';
import { FIXTURE_DECODERS } from '../fixtures/decoders';
import { homePin } from '../state/atlas';
import { buildRequest, capsContext } from '../state/requests';
import { layerSpans } from '../state/resolve';
import type { Speaker } from './api';
import { compileScript, NO_CONTROLS, type ScriptEnv } from './script';

const spec = fixture as unknown as ControlsSpec;
const speaker = (id: string, label: string, speaker_modes: Speaker['speaker_modes'] = ['embedding', 'reference']) =>
  ({ id, label, scope: 'default', is_default: true, source_type: 'audio', speaker_modes, has_preview: true, embedding_dim: 192, transcript: null, disabled_controls: [], original_name: null, created_at: 0 }) as Speaker;
const ann = speaker('ann', 'Ann en British F 1');
const lie = speaker('lie', 'Liesel de F 1');
const ref = speaker('ref', 'Ref', ['reference']);
const caps: ScriptEnv['caps'] = {
  decoders: FIXTURE_DECODERS,
  laci_languages: ['en_us'],
  phoneme_tokens: null,
  start_token: null,
  pause_tokens: null,
};
const env: ScriptEnv = {
  spec,
  steering: true,
  speakers: [ann, lie, ref],
  caps,
  languages: ['en_us', 'en_gb', 'de'],
  selected: ann,
  selectedWhy: null,
  language: 'en_us',
  seed: 7,
  laci: 'off',
};
const compile = (text: string, over: Partial<ScriptEnv> = {}) => compileScript(text, { ...env, ...over });
const messages = (text: string, over: Partial<ScriptEnv> = {}) => compile(text, over).errors.map((e) => `${e.line}: ${e.message}`);

describe('lite script → requests', () => {
  it('sends what Studio sends for the same design, baseline and painting', () => {
    const r = compile('@baseline pace -0.5\n@voice A = ann {age 1}\nA: Welcome to [the late show]{projection 1.5}.');
    expect(r.errors).toEqual([]);
    const ctx = capsContext(caps, { decoder: null, laci: 'off' });
    const spans = layerSpans(25, null, [{ start: 11, end: 24, values: { projection: 1.5 } }]);
    const studio = buildRequest(spec, ctx, { baseId: 'ann', mode: 'embedding', values: { age: 1 }, pin: homePin() }, 'Welcome to the late show.', 7, { extra: { pace: -0.5 }, spans, language: 'en_us' });
    expect(r.render!.parts).toHaveLength(1);
    expect(r.render!.parts[0]).toMatchObject({ line: 3, label: 'A', text: 'Welcome to the late show.' });
    expect(r.render!.parts[0].request).toEqual({ ...studio, stream: true, format: 'pcm' });
    expect(r.render!.parts[0].request.control_spans![0]).toMatchObject({ start: 11, end: 24 });
  });

  it('waits for step 1’s clip to be cloned, and says why step 1 has no voice', () => {
    const waiting = compile('Hello there.', { selected: null });
    expect(waiting).toMatchObject({ errors: [], usesSelected: true, render: null });
    expect(messages('# intro\nHello there.', { selected: null, selectedWhy: 'Pick a voice first' })).toEqual(['2: This line speaks with the voice of step 1: Pick a voice first.']);
  });

  it('puts each error on its line', () => {
    expect(messages('Ann {gender 1}: Hi')).toEqual(["1: 'gender' is a voice control, not a delivery control: set it on the voice."]);
    expect(messages('{emphasis 4}: Hi')).toEqual(['1: Emphasis goes on words, not a whole line: [words]{emphasis 4}.']);
    expect(messages('@voice R = Ref {deep 2}\nR: Hi')).toEqual(['1: Ref has no embedding mode.']);
    expect(messages('@voice R = ann {deep 2, mode reference}\nR: Hi')).toEqual(['1: Deep needs mode embedding (mode reference keeps the recording: only delivery and mood apply).']);
    expect(messages('@voice B = ann {mode reference, preset BBC newsreader}\nB: Hi')).toEqual(['1: A preset sets mode embedding.']);
    expect(messages('@seed -1\nHi')).toEqual(['1: @seed takes a whole number from 0 to 4294967295.']);
    expect(messages('Hi\n@gap 2')).toEqual(['2: @gap takes seconds from 0 to 1.5.']);
    expect(messages('@foo\nHi')).toEqual(['1: Unknown directive @foo. Directives: @language, @mode, @seed, @laci, @decoder, @gap, @baseline, @start, @voice.']);
    expect(messages('@mode embedding\n@mode reference\nHi')).toEqual(['2: @mode is set twice (first on line 1).']);
  });

  it('lets @mode pick only for voices without a design', () => {
    const r = compile('@mode reference\n@voice D {deep 2}\nD: Hi');
    expect(r.errors).toEqual([]);
    expect(r.render!.parts[0].request.speaker_mode).toBe('embedding');
  });

  it('puts each error on the characters it is about', () => {
    const slice = (t: string) => {
      const e = compile(t).errors[0];
      return t.slice(e.from, e.to);
    };
    expect(slice('Ann {gender 1}: Hi')).toBe('gender 1');
    expect(slice('@foo\nHi')).toBe('@foo');
    expect(slice('Hi [ ]{happy 1}')).toBe('[ ]{happy 1}');
  });

  it('takes @seed from its line on over Advanced’s seed, and @language over a default voice’s own language', () => {
    const seeds = (t: string, over: Partial<ScriptEnv> = {}) => compile(t, over).render!.parts.map((p) => p.request.seed);
    expect(seeds('Hi\n@seed 3\nHo\n@seed 4\nHa', { seed: 9 })).toEqual([9, 3, 4]);
    expect(compile('@language en_gb\nHi.\nLiesel de F 1: Hallo.').render!.parts.map((p) => p.request.language)).toEqual(['en_gb', 'en_gb']);
    expect(compile('Hi.\nLiesel de F 1: Hallo.').render!.parts.map((p) => p.request.language ?? 'en_us')).toEqual(['en_us', 'de']);
  });

  it('turns LACI on where the server has heads, and says where it has none', () => {
    const r = compile('@laci whole\nHi.\nlie: Hallo.');
    expect(r.render!.parts[0].request).toMatchObject({ laci: true, stream_decode: 'whole' });
    expect(r.render!.parts[1].request.laci).toBeUndefined();
    expect(r.warnings).toEqual([{ line: 3, from: 21, to: 27, message: 'No alignment heads for German: its lines run without LACI.' }]);
  });

  it('renders plain lines without a steering bundle and refuses controls', () => {
    const off = { steering: false, spec: NO_CONTROLS };
    const r = compile('Hi there.', off);
    expect(r.errors).toEqual([]);
    expect(r.render!.parts[0].request.controls).toBeUndefined();
    expect(messages('{happy 2}: hi', off)).toEqual(['1: This server has no steering bundle (GET /v2/controls answers 404): controls, presets and accents are off.']);
  });

  it('reads a backslash line as text, and warns about a name that is no voice', () => {
    const r = compile('\\@home\n\\Ann: hi\nNote: hi');
    expect(r.errors).toEqual([]);
    expect(r.render!.parts.map((p) => [p.text, p.label])).toEqual([
      ['@home', 'Ann en British F 1'],
      ['Ann: hi', 'Ann en British F 1'],
      ['Note: hi', 'Ann en British F 1'],
    ]);
    expect(r.warnings).toEqual([{ line: 3, from: 16, to: 20, message: '“Note:” is spoken: no voice is named Note.' }]);
    expect(compile('Ann: hi').render!.parts[0].text).toBe('hi');
  });
});

const phonemeTokens: PronTokens = phonemes.phoneme_tokens;
const pauseTokens = { markers: ['\ue003', '\ue004', '\ue005', '\ue006'], seconds: [0.2475, 0.495, 0.9899, 1.9799] };
const fullCaps: ScriptEnv['caps'] = {
  decoders: FIXTURE_DECODERS,
  laci_languages: ['en_us', 'en_gb', 'fr_fr'],
  phoneme_tokens: phonemeTokens,
  start_token: { marker: '\ue002' },
  pause_tokens: pauseTokens,
};

describe('the Night Ferry script', () => {
  const library = ['Imogen en British F 1', 'Cillian en Irish M 1', 'Savannah en American F 1', 'Gaspard fr M 1'].map((label, i) => speaker(`spk${i}`, label));
  const mine = { ...speaker('mine', 'Mine'), scope: 'session', is_default: false } as Speaker;
  const ferryEnv: Partial<ScriptEnv> = { speakers: [...library, mine], selected: mine, languages: ['en_us', 'en_gb', 'fr_fr'], caps: fullCaps };

  it('compiles, with each directive in force from its line on', () => {
    const r = compile(nightFerry, ferryEnv);
    expect(r.errors).toEqual([]);
    const parts = r.render!.parts;
    const byLine = (n: number) => parts.find((p) => p.line === n)!;
    expect(parts).toHaveLength(51);
    expect(byLine(21).request).toEqual(byLine(19).request);
    expect(byLine(23).request.seed).toBe(7);
    expect(byLine(19).request.decoder).toBe('full_mr');
    expect(byLine(28).request.decoder).toBeUndefined();
    expect(byLine(19).request.laci).toBeUndefined();
    expect(byLine(36).request).toMatchObject({ laci: true, stream_decode: 'whole' });
    expect([36, 37, 43, 44, 75, 68].map((n) => byLine(n).request.speaker_mode)).toEqual(['reference', 'embedding', 'reference', 'embedding', 'embedding', 'reference']);
    expect(byLine(50).request.text).toContain('\ue000ʃəvɒn\ue001');
    expect(byLine(50).request.text).toContain('\ue000kəʊv\ue001');
    expect([68, 115, 117].map((n) => byLine(n).request.language)).toEqual(['en_gb', 'fr_fr', 'en_gb']);
    expect([92, 93, 98].map((n) => byLine(n).gap)).toEqual([1.5, 1.5, 0.5]);
    expect(parts[0].gap).toBe(0);
  });

  it('keeps mode reference with a preset an error', () => {
    const lines = nightFerry.split('\n');
    lines[7] = '@voice Ann = Imogen en British F 1 {preset BBC newsreader, age 1, accent irish 1.2, mode reference}';
    expect(messages(lines.join('\n'), ferryEnv)).toEqual(['8: A preset sets mode embedding.']);
  });
});

describe('start and pauses', () => {
  const full = { caps: fullCaps };

  it('starts immediately per line and from @start on', () => {
    const r = compile('{start}: Hi.\n@start on\nHo.\n@start off\nHa.', full);
    expect(r.errors).toEqual([]);
    expect(r.render!.parts.map((p) => p.request.start_immediately)).toEqual([true, true, undefined]);
    expect(messages('{start}: Hi.')).toEqual(['1: This server has no start token.']);
  });

  it('sends a pause mark as the nearest pause token, and refuses it without them', () => {
    expect(compile('Wait, <0.5> then go.', full).render!.parts[0].request.text).toBe('Wait,\ue004 then go.');
    expect(messages('Wait, <0.5> then go.')).toEqual(['1: Pauses: “<0.5>” at 7: this server has no pause tokens.']);
  });

  it('paints a span over a pronunciation group', () => {
    const r = compile('I met [[[S@vQn]]]{emphasis 4}.', full);
    expect(r.errors).toEqual([]);
    const req = r.render!.parts[0].request;
    expect(req.control_spans).toHaveLength(1);
    expect(req.control_spans![0]).toMatchObject({ start: req.text.indexOf('\ue000'), end: req.text.indexOf('\ue001') + 1 });
  });
});
