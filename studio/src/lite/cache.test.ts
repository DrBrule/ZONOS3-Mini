import { describe, expect, it } from 'vitest';
import { LineCache } from './cache';

const pcm = (n: number) => new Float32Array(n).fill(n);

describe('LineCache', () => {
  it('drops the least recently used line past its size', () => {
    const c = new LineCache(10);
    c.put('a', pcm(6));
    c.put('b', pcm(6));
    expect(c.get('a')).toBeUndefined();
    expect(c.get('b')).toEqual(pcm(6));
  });

  it('keeps a line it was just asked for', () => {
    const c = new LineCache(10);
    c.put('a', pcm(4));
    c.put('b', pcm(4));
    c.get('a');
    c.put('c', pcm(4));
    expect(c.get('b')).toBeUndefined();
    expect(c.get('a')).toEqual(pcm(4));
    expect(c.get('c')).toEqual(pcm(4));
  });

  it('ignores empty audio and audio larger than the cache', () => {
    const c = new LineCache(10);
    c.put('a', pcm(4));
    c.put('empty', pcm(0));
    c.put('big', pcm(11));
    expect(c.get('empty')).toBeUndefined();
    expect(c.get('big')).toBeUndefined();
    expect(c.get('a')).toEqual(pcm(4));
  });
});
