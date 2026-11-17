import { describe, expect, it } from 'vitest';
import { parsePron, spokenText, textProblem, toSent, type PauseTokens } from './pron';

const P: PauseTokens = { markers: ['\ue003', '\ue004', '\ue005', '\ue006'], seconds: [0.2475, 0.495, 0.9899, 1.9799] };

describe('pronunciation groups and pause marks', () => {
  it('leaves a third bracket outside the group', () => {
    expect(parsePron('a [[[S@vQn]]]{x}', null).groups).toMatchObject([{ start: 3, end: 12, body: 'S@vQn' }]);
  });

  it('refuses pauses in speech edits', () => {
    expect(textProblem('Wait, <0.5> go', { phonemes: null, pauses: P }, 'edit')).toBe('Pauses: “<0.5>” at 7: speech edits take no pauses.');
  });

  it('sends a pause mark, with the space before it, as the nearest pause token', () => {
    expect(toSent('a, <0.75> b', { phonemes: null, pauses: P }).text).toBe('a,\ue005 b');
    expect(toSent('a <laugh/> b', { phonemes: null, pauses: P }).text).toBe('a <laugh/> b');
    expect(spokenText('a, <1s> b')).toBe('a, b');
  });
});
