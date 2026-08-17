/**
 * Reference backend client for the hosted all-in-one Glimms deployment.
 *
 *   const glimms = new GlimmsClient({
 *     baseUrl: process.env.GLIMMS_BASE_URL!,
 *     internalToken: process.env.GLIMMS_INTERNAL_TOKEN,
 *   });
 *
 * Server-side only. It carries the shared internal token, so it must never be
 * bundled into browser or mobile code.
 */

export interface GlimmsOptions {
  baseUrl: string;
  /** Sent as `Authorization: Bearer ...`. Required unless the deployment is
   *  running with AI_INTERNAL_TOKEN unset (development only). */
  internalToken?: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** Max simultaneous in-flight requests. One container runs all 8 services. */
  concurrency?: number;
}

export type Vertical = "wardrobe" | "room" | "garden";

export class GlimmsError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly service: string,
    readonly correlationId: string,
    readonly retryable: boolean,
  ) {
    super(message);
    this.name = "GlimmsError";
  }
}

/** Minimal counting semaphore so one job cannot saturate the instance. */
class Semaphore {
  private active = 0;
  private queue: Array<() => void> = [];
  constructor(private readonly limit: number) {}

  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (this.active >= this.limit) {
      await new Promise<void>((resolve) => this.queue.push(resolve));
    }
    this.active += 1;
    try {
      return await fn();
    } finally {
      this.active -= 1;
      this.queue.shift()?.();
    }
  }
}

const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class GlimmsClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly gate: Semaphore;

  constructor(opts: GlimmsOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.token = opts.internalToken;
    this.timeoutMs = opts.timeoutMs ?? 120_000;
    this.maxRetries = opts.maxRetries ?? 3;
    this.gate = new Semaphore(opts.concurrency ?? 6);
  }

  // ---------------------------------------------------------------- transport

  private async call<T>(
    service: string,
    path: string,
    body: unknown,
    correlationId: string,
  ): Promise<T> {
    const url = `${this.baseUrl}/${service}${path}`;
    let lastError: GlimmsError | null = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const response = await this.gate.run(() =>
          fetch(url, {
            method: "POST",
            signal: controller.signal,
            headers: {
              "content-type": "application/json",
              "x-correlation-id": correlationId,
              ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
            },
            body: JSON.stringify(body),
          }),
        );

        if (response.ok) return (await response.json()) as T;

        const detail = await response.text().catch(() => "");

        // A blocked development fallback is a configuration problem, not a
        // transient one: retrying it just burns the job's budget.
        const blocked =
          response.status === 503 && detail.includes("development_fallback_blocked");

        lastError = new GlimmsError(
          `${service}${path} failed: ${response.status} ${detail.slice(0, 300)}`,
          response.status,
          service,
          correlationId,
          !blocked && RETRYABLE_STATUS.has(response.status),
        );
        if (!lastError.retryable) throw lastError;
      } catch (err) {
        if (err instanceof GlimmsError) {
          if (!err.retryable) throw err;
          lastError = err;
        } else {
          // Network reset / abort / Render cold-start hang: worth retrying.
          lastError = new GlimmsError(
            `${service}${path} transport error: ${(err as Error).message}`,
            null,
            service,
            correlationId,
            true,
          );
        }
      } finally {
        clearTimeout(timer);
      }

      if (attempt < this.maxRetries) {
        const backoff = 500 * 2 ** attempt;
        await sleep(backoff + Math.random() * backoff);
      }
    }

    throw lastError!;
  }

  // ------------------------------------------------------------------- health

  async health(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`, {
      signal: AbortSignal.timeout(30_000),
      headers: this.token ? { authorization: `Bearer ${this.token}` } : {},
    });
    if (!response.ok) {
      throw new GlimmsError("health check failed", response.status, "gateway", "-", true);
    }
    return response.json();
  }

  /**
   * True when the deployment is serving deterministic development fallbacks
   * instead of real models. The gateway computes this itself; the local
   * fallback below keeps the client working against older deployments.
   */
  async isDegraded(): Promise<boolean> {
    const h = await this.health();
    if (typeof h.production_ready === "boolean") return !h.production_ready;

    const s = h.services ?? {};
    return (
      s["object-detection"]?.model_loaded === false ||
      s["attribute-extractor"]?.clip_enabled === false ||
      s["embedding-engine"]?.backend === "memory" ||
      !s["llm-reasoning"]?.providers_configured?.some((p: string) => p !== "free-fallback")
    );
  }

  /**
   * Readiness as the deployment itself judges it: 200 only when every service
   * is reachable and none is answering with a blocked development fallback.
   */
  async isReady(): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/readyz`, {
      signal: AbortSignal.timeout(30_000),
      headers: this.token ? { authorization: `Bearer ${this.token}` } : {},
    });
    return response.ok;
  }

  // ----------------------------------------------------------------- services

  qualityCheck(imageKeys: string[], cid: string) {
    return this.call<any>("quality-guard", "/check", { image_keys: imageKeys }, cid);
  }

  detect(imageKeys: string[], vertical: Vertical, cid: string) {
    return this.call<any>(
      "object-detection",
      "/detect",
      { image_keys: imageKeys, vertical },
      cid,
    );
  }

  extract(items: Array<Record<string, unknown>>, cid: string) {
    return this.call<any>("attribute-extractor", "/extract", { items }, cid);
  }

  inferContext(input: Record<string, unknown>, cid: string) {
    return this.call<any>("context-inference", "/infer", input, cid);
  }

  permute(
    input: {
      items: Array<Record<string, unknown>>;
      vertical: Vertical;
      context?: Record<string, unknown>;
      max_permutations?: number;
    },
    cid: string,
  ) {
    return this.call<any>("permutation-engine", "/generate", input, cid);
  }

  upsertVectors(
    vectors: Array<{ id: string; embedding: number[]; metadata?: Record<string, unknown> }>,
    cid: string,
    namespace?: string,
  ) {
    return this.call<any>("embedding-engine", "/upsert", { vectors, namespace }, cid);
  }

  searchVectors(embedding: number[], cid: string, topK = 10, namespace?: string) {
    return this.call<any>(
      "embedding-engine",
      "/search",
      { embedding, top_k: topK, namespace },
      cid,
    );
  }

  reason(
    input: {
      vertical: Vertical;
      context: Record<string, unknown>;
      permutations: Array<Record<string, unknown>>;
    },
    cid: string,
  ) {
    return this.call<any>("llm-reasoning", "/reason", input, cid);
  }

  compose(
    input: {
      layers: Array<{ image_key: string; bbox?: Record<string, number> }>;
      output_key?: string;
      width?: number;
      height?: number;
      format?: "jpg" | "png";
      background?: string;
    },
    cid: string,
  ) {
    return this.call<any>("mockup-compositor", "/compose", input, cid);
  }
}

/**
 * One pipeline run. Call this from a queue worker, never from an HTTP handler:
 * on Render's shared tier a cold start alone can exceed a request timeout.
 * Persist each step's result so a failure can resume instead of restarting.
 */
export async function runPipeline(
  glimms: GlimmsClient,
  session: {
    id: string;
    vertical: Vertical;
    imageKeys: string[];
    context: Record<string, unknown>;
  },
  persist: (step: string, payload: unknown) => Promise<void>,
) {
  const cid = `cor_${session.id}`;

  const [quality, detections, context] = await Promise.all([
    glimms.qualityCheck(session.imageKeys, cid),
    glimms.detect(session.imageKeys, session.vertical, cid),
    glimms.inferContext({ vertical: session.vertical, ...session.context }, cid),
  ]);
  await Promise.all([
    persist("quality", quality),
    persist("detections", detections),
    persist("context", context),
  ]);

  if (!quality.passed) {
    // Surface the per-image `guidance` to the user and stop: bad inputs
    // produce confidently bad designs.
    return { status: "needs_recapture" as const, quality };
  }

  const attributes = await glimms.extract(detections.items, cid);
  await persist("attributes", attributes);

  const permutations = await glimms.permute(
    {
      items: attributes.items,
      vertical: session.vertical,
      context: context.style_constraints ?? context,
      max_permutations: 20,
    },
    cid,
  );
  await persist("permutations", permutations);

  const designs = await glimms.reason(
    {
      vertical: session.vertical,
      context: context.style_constraints ?? context,
      permutations: permutations.permutations,
    },
    cid,
  );
  await persist("designs", designs);

  return { status: "complete" as const, designs, quality };
}
