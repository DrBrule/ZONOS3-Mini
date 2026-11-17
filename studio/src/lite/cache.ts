/** Finished script lines' audio by request, least recently used out first past `maxSamples`. */
export class LineCache {
  private readonly lines = new Map<string, Float32Array>();
  private total = 0;

  constructor(private readonly maxSamples: number) {}

  /** The audio of `key`; a hit becomes the most recent. */
  get(key: string): Float32Array | undefined {
    const pcm = this.lines.get(key);
    if (pcm) {
      this.lines.delete(key);
      this.lines.set(key, pcm);
    }
    return pcm;
  }

  /** Keeps `pcm` under `key` (replacing what it held); empty audio and audio longer than the whole cache are not kept. */
  put(key: string, pcm: Float32Array): void {
    if (!pcm.length || pcm.length > this.maxSamples) return;
    const old = this.lines.get(key);
    if (old) {
      this.lines.delete(key);
      this.total -= old.length;
    }
    this.lines.set(key, pcm);
    this.total += pcm.length;
    for (const [k, v] of this.lines) {
      if (this.total <= this.maxSamples) break;
      this.lines.delete(k);
      this.total -= v.length;
    }
  }
}
