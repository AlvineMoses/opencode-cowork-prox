import { Hono } from 'hono';
import {
  extractApiKey,
  validateApiKey,
  authErrorResponse,
} from './auth';

import { formatAnthropicToOpenAI } from './translate/request/anthropic-to-openai';
import { formatOpenAIToAnthropic } from './translate/request/openai-to-anthropic';

import { formatOpenAIToAnthropic as toAnthropicResponse } from './translate/response/openai-to-anthropic';
import { formatAnthropicToOpenAI as toOpenAIResponse } from './translate/response/anthropic-to-openai';

import { streamOpenAIToAnthropic } from './translate/stream/openai-to-anthropic';
import { streamAnthropicToOpenAI } from './translate/stream/anthropic-to-openai';

const GO_UPSTREAM = 'https://opencode.ai/zen/go/v1';
const ZEN_UPSTREAM = 'https://opencode.ai/zen/v1';
const DEFAULT_UPSTREAM = GO_UPSTREAM;

const API_START_PATHS = new Set(['v1', 'v2']);

type UpstreamProtocol = 'openai' | 'anthropic' | 'responses';

type RouteConfig = {
  path: string;
  upstream: string;
  modelOverride: string | null;
};

type ModelInfo = {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
  [key: string]: unknown;
};

/**
 * Claude Desktop requires gateway model IDs that look like
 * Anthropic models.
 *
 * These are VIRTUAL Claude IDs.
 *
 * Claude Desktop sees:
 *
 *   anthropic/claude-opencode-glm-5-2
 *
 * The Worker internally sends:
 *
 *   glm-5.2
 *
 * to OpenCode.
 *
 * The aliases below also include Claude Desktop's built-in
 * credential-probe model.
 */
const MODEL_ALIASES: Record<string, string> = {
  /*
   * Claude Desktop gateway credential probe.
   *
   * IMPORTANT:
   * Claude Desktop has been observed probing both of these.
   */
  'sonnet-5.1': 'glm-5.2',
  'sonnet5.1': 'glm-5.2',
  'claude-sonnet-4-5': 'glm-5.2',

  /*
   * Claude-compatible virtual routes.
   */
  'anthropic/claude-opencode-glm-5-2': 'glm-5.2',
  'anthropic/claude-opencode-glm-5-1': 'glm-5.1',

  'anthropic/claude-opencode-grok-4-5': 'grok-4.5',

  'anthropic/claude-opencode-kimi-k3': 'kimi-k3',
  'anthropic/claude-opencode-kimi-k2-7-code': 'kimi-k2.7-code',
  'anthropic/claude-opencode-kimi-k2-6': 'kimi-k2.6',

  'anthropic/claude-opencode-deepseek-v4-pro':
    'deepseek-v4-pro',
  'anthropic/claude-opencode-deepseek-v4-flash':
    'deepseek-v4-flash',

  'anthropic/claude-opencode-mimo-v2-5':
    'mimo-v2.5',
  'anthropic/claude-opencode-mimo-v2-5-pro':
    'mimo-v2.5-pro',

  'anthropic/claude-opencode-hy3':
    'hy3',

  'anthropic/claude-opencode-minimax-m3':
    'minimax-m3',
  'anthropic/claude-opencode-minimax-m2-7':
    'minimax-m2.7',
  'anthropic/claude-opencode-minimax-m2-5':
    'minimax-m2.5',

  'anthropic/claude-opencode-qwen3-8-max':
    'qwen3.8-max',
  'anthropic/claude-opencode-qwen3-7-max':
    'qwen3.7-max',
  'anthropic/claude-opencode-qwen3-7-plus':
    'qwen3.7-plus',
  'anthropic/claude-opencode-qwen3-6-plus':
    'qwen3.6-plus',
};

/**
 * Upstream API protocol for each OpenCode model.
 */
const MODEL_PROTOCOLS: Record<string, UpstreamProtocol> = {
  'grok-4.5': 'openai',

  'glm-5.2': 'openai',
  'glm-5.1': 'openai',

  'kimi-k3': 'openai',
  'kimi-k2.7-code': 'openai',
  'kimi-k2.6': 'openai',

  'deepseek-v4-pro': 'openai',
  'deepseek-v4-flash': 'openai',

  'mimo-v2.5': 'openai',
  'mimo-v2.5-pro': 'openai',

  'hy3': 'openai',

  'minimax-m3': 'anthropic',
  'minimax-m2.7': 'anthropic',
  'minimax-m2.5': 'anthropic',

  'qwen3.8-max': 'anthropic',
  'qwen3.7-max': 'anthropic',
  'qwen3.7-plus': 'anthropic',
  'qwen3.6-plus': 'anthropic',

  /*
   * Not enabled yet because this Worker does not have a
   * Responses API adapter.
   */
  'gpt-5.6-luna': 'responses',
};

function stripPrefix(
  path: string,
  prefix: string,
): string | null {
  if (path === prefix) return '/';

  if (path.startsWith(`${prefix}/`)) {
    return path.slice(prefix.length);
  }

  return null;
}

function extractModelSegment(path: string): {
  path: string;
  model: string | null;
} {
  const segments = path.replace(/^\/+/, '').split('/');

  if (
    segments.length > 0 &&
    segments[0] &&
    !API_START_PATHS.has(segments[0])
  ) {
    return {
      path: '/' + segments.slice(1).join('/'),
      model: segments[0],
    };
  }

  return {
    path,
    model: null,
  };
}

function routeConfig(request: Request): RouteConfig {
  const path = new URL(request.url).pathname;

  const goPath = stripPrefix(path, '/go');

  if (goPath !== null) {
    const {
      path: remaining,
      model,
    } = extractModelSegment(goPath);

    return {
      path: remaining,
      upstream: GO_UPSTREAM,
      modelOverride: model,
    };
  }

  const zenPath = stripPrefix(path, '/zen');

  if (zenPath !== null) {
    const {
      path: remaining,
      model,
    } = extractModelSegment(zenPath);

    return {
      path: remaining,
      upstream: ZEN_UPSTREAM,
      modelOverride: model,
    };
  }

  const {
    path: remaining,
    model,
  } = extractModelSegment(path);

  return {
    path: remaining,
    upstream: DEFAULT_UPSTREAM,
    modelOverride: model,
  };
}

function getUpstream(
  request: Request,
  routeUpstream: string,
): string {
  return (
    request.headers.get('X-Upstream-Url') ||
    routeUpstream
  );
}

function upstreamFormat(
  request: Request,
): 'openai' | 'anthropic' {
  const fmt = (
    request.headers.get('X-Upstream-Format') ||
    'openai'
  ).toLowerCase();

  return fmt === 'anthropic'
    ? 'anthropic'
    : 'openai';
}

function anthropicHeaders(
  request: Request,
  key: string,
): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Api-Key': key,
    'Anthropic-Version':
      request.headers.get('Anthropic-Version') ||
      '2023-06-01',
  };

  const beta = request.headers.get('Anthropic-Beta');

  if (beta) {
    headers['Anthropic-Beta'] = beta;
  }

  return headers;
}

function openAIHeaders(
  key: string,
): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${key}`,
  };
}

/**
 * Converts a Claude/Desktop-facing model ID into the actual
 * OpenCode model ID.
 */
function normalizeModelId(
  model: unknown,
): string | null {
  if (typeof model !== 'string') {
    return null;
  }

  let normalized = model.trim();

  if (normalized.startsWith('opencode-go/')) {
    normalized = normalized.slice(
      'opencode-go/'.length,
    );
  }

  if (normalized.startsWith('opencode/')) {
    normalized = normalized.slice(
      'opencode/'.length,
    );
  }

  return (
    MODEL_ALIASES[normalized] ||
    normalized
  );
}

function getModelProtocol(
  model: string | null,
): UpstreamProtocol {
  if (!model) {
    return 'openai';
  }

  const normalized =
    normalizeModelId(model);

  if (!normalized) {
    return 'openai';
  }

  return (
    MODEL_PROTOCOLS[normalized] ||
    'openai'
  );
}

function hasImages(body: any): boolean {
  const messages = body?.messages;

  if (!Array.isArray(messages)) {
    return false;
  }

  return messages.some(
    (msg: any) =>
      Array.isArray(msg.content) &&
      msg.content.some(
        (part: any) =>
          part?.type === 'image',
      ),
  );
}

function upstreamErrorResponse(
  res: Response,
  body: string,
): Response {
  const headers = new Headers();

  for (
    const name of [
      'Content-Type',
      'Retry-After',
      'RateLimit-Limit',
      'RateLimit-Remaining',
      'RateLimit-Reset',
    ]
  ) {
    const value =
      res.headers.get(name);

    if (value) {
      headers.set(name, value);
    }
  }

  return new Response(body, {
    status: res.status,
    headers,
  });
}

/**
 * Convert an OpenCode model ID into the Claude-compatible
 * virtual ID exposed to Claude Desktop.
 */
function toClaudeModelId(
  modelId: string,
): string {
  const existing =
    Object.entries(MODEL_ALIASES)
      .find(
        ([alias, upstream]) =>
          alias.startsWith(
            'anthropic/claude-opencode-',
          ) &&
          upstream === modelId,
      );

  if (existing) {
    return existing[0];
  }

  const safe = modelId
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();

  return `anthropic/claude-opencode-${safe}`;
}

/**
 * Return the OpenCode Go catalog as a Claude-compatible
 * Anthropic-looking model catalog.
 */
async function fetchGoModels(
  key: string,
): Promise<Response> {
  const res = await fetch(
    `${GO_UPSTREAM}/models`,
    {
      method: 'GET',
      headers: {
        Authorization:
          `Bearer ${key}`,
      },
    },
  );

  if (!res.ok) {
    return upstreamErrorResponse(
      res,
      await res.text(),
    );
  }

  const body: any =
    await res.json();

  const upstreamModels: ModelInfo[] =
    Array.isArray(body?.data)
      ? body.data
      : [];

  /*
   * Only expose models for which this Worker actually knows
   * how to communicate with the upstream API.
   */
  const supportedModels =
    upstreamModels.filter(
      (model) => {
        const normalized =
          normalizeModelId(model.id);

        return (
          normalized !== null &&
          MODEL_PROTOCOLS[
            normalized
          ] !== undefined
        );
      },
    );

  const models =
    supportedModels.map(
      (model) => {
        const normalized =
          normalizeModelId(model.id)!;

        return {
          id: toClaudeModelId(
            normalized,
          ),

          object:
            model.object ||
            'model',

          created:
            model.created ||
            Math.floor(
              Date.now() / 1000,
            ),

          owned_by:
            'anthropic',

          /*
           * Keep the real OpenCode model available
           * for debugging/UI metadata.
           */
          openCodeModel:
            normalized,

          /*
           * Helpful display metadata.
           */
          display_name:
            `OpenCode — ${normalized}`,

          provider:
            'opencode-go',
        };
      },
    );

  /*
   * Always include the credential-probe model.
   *
   * This is important because Claude Desktop may request
   * this model before loading the actual model picker.
   */
  const probeModel = {
    id: 'claude-sonnet-4-5',
    object: 'model',
    created: Math.floor(
      Date.now() / 1000,
    ),
    owned_by: 'anthropic',
    openCodeModel: 'glm-5.2',
    display_name:
      'Claude Sonnet 4.5 (OpenCode)',
    provider: 'opencode-go',
  };

  const hasProbe =
    models.some(
      (model) =>
        model.id ===
        probeModel.id,
    );

  if (!hasProbe) {
    models.unshift(
      probeModel,
    );
  }

  return new Response(
    JSON.stringify({
      object: 'list',
      data: models,
    }),
    {
      status: 200,
      headers: {
        'Content-Type':
          'application/json',
      },
    },
  );
}

/**
 * Claude Desktop -> Anthropic Messages -> OpenCode.
 */
async function handleAnthropicMessages(
  request: Request,
  route: RouteConfig,
  upstream: string,
  key: string,
): Promise<Response> {
  const req =
    await request.json();

  /*
   * Keep the original Claude-facing ID only for
   * response metadata.
   */
  const originalModel =
    typeof req.model === 'string'
      ? req.model
      : null;

  /*
   * Convert Claude virtual model ID into
   * actual OpenCode model.
   */
  const selectedModel =
    route.modelOverride ||
    req.model;

  const upstreamModel =
    normalizeModelId(
      selectedModel,
    );

  if (!upstreamModel) {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            'No model was supplied.',
        },
      }),
      {
        status: 400,
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  /*
   * IMPORTANT:
   * The translator needs to send the real OpenCode model.
   */
  req.model =
    upstreamModel;

  const protocol =
    getModelProtocol(
      upstreamModel,
    );

  if (protocol === 'openai') {
    const openaiReq =
      formatAnthropicToOpenAI(
        req,
      );

    const res =
      await fetch(
        `${upstream}/chat/completions`,
        {
          method: 'POST',
          headers:
            openAIHeaders(key),
          body:
            JSON.stringify(
              openaiReq,
            ),
        },
      );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    if (openaiReq.stream) {
      return new Response(
        streamOpenAIToAnthropic(
          res.body as ReadableStream,
          originalModel ||
            upstreamModel,
        ),
        {
          headers: {
            'Content-Type':
              'text/event-stream',
            'Cache-Control':
              'no-cache',
            Connection:
              'keep-alive',
          },
        },
      );
    }

    const data: any =
      await res.json();

    return new Response(
      JSON.stringify(
        toAnthropicResponse(
          data,
          originalModel ||
            upstreamModel,
        ),
      ),
      {
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  if (protocol === 'anthropic') {
    const res =
      await fetch(
        `${upstream}/messages`,
        {
          method: 'POST',
          headers:
            anthropicHeaders(
              request,
              key,
            ),
          body:
            JSON.stringify(req),
        },
      );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    return new Response(
      res.body,
      {
        status: res.status,
        headers: res.headers,
      },
    );
  }

  if (protocol === 'responses') {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            `Model ${upstreamModel} uses the OpenAI Responses API and is not yet supported by this compatibility layer.`,
        },
      }),
      {
        status: 501,
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  return new Response(
    JSON.stringify({
      type: 'error',
      error: {
        type: 'ModelError',
        message:
          `Unsupported upstream protocol for model ${upstreamModel}.`,
      },
    }),
    {
      status: 400,
      headers: {
        'Content-Type':
          'application/json',
      },
    },
  );
}

async function handleRequest(
  request: Request,
): Promise<Response> {
  const route =
    routeConfig(request);

  const upstream =
    getUpstream(
      request,
      route.upstream,
    );

  const fmt =
    upstreamFormat(request);

  /*
   * ============================================================
   * Anthropic Messages
   * ============================================================
   */
  if (
    route.path === '/v1/messages' &&
    request.method === 'POST'
  ) {
    const key =
      extractApiKey(
        request.headers,
      );

    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(err);
    }

    /*
     * OpenCode Go through our model router.
     */
    if (
      fmt === 'openai' &&
      upstream === GO_UPSTREAM
    ) {
      return handleAnthropicMessages(
        request,
        route,
        upstream,
        key!,
      );
    }

    /*
     * Generic Anthropic upstream.
     */
    if (
      fmt === 'anthropic'
    ) {
      const body =
        await request.text();

      return fetch(
        `${upstream}/messages`,
        {
          method: 'POST',
          headers:
            anthropicHeaders(
              request,
              key!,
            ),
          body,
        },
      );
    }

    /*
     * Generic OpenAI-compatible upstream.
     */
    const req =
      await request.json();

    const originalModel =
      req.model;

    if (route.modelOverride) {
      req.model =
        route.modelOverride;
    }

    const openaiReq =
      formatAnthropicToOpenAI(
        req,
      );

    const res =
      await fetch(
        `${upstream}/chat/completions`,
        {
          method: 'POST',
          headers:
            openAIHeaders(
              key!,
            ),
          body:
            JSON.stringify(
              openaiReq,
            ),
        },
      );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    if (openaiReq.stream) {
      return new Response(
        streamOpenAIToAnthropic(
          res.body as ReadableStream,
          originalModel,
        ),
        {
          headers: {
            'Content-Type':
              'text/event-stream',
            'Cache-Control':
              'no-cache',
            Connection:
              'keep-alive',
          },
        },
      );
    }

    const data: any =
      await res.json();

    return new Response(
      JSON.stringify(
        toAnthropicResponse(
          data,
          originalModel,
        ),
      ),
      {
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  /*
   * ============================================================
   * OpenAI Chat Completions
   * ============================================================
   */
  if (
    route.path ===
      '/v1/chat/completions' &&
    request.method === 'POST'
  ) {
    const key =
      extractApiKey(
        request.headers,
      );

    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(err);
    }

    if (
      fmt === 'anthropic'
    ) {
      const req =
        await request.json();

      const anthReq =
        formatOpenAIToAnthropic(
          req,
        );

      const res =
        await fetch(
          `${upstream}/messages`,
          {
            method: 'POST',
            headers:
              anthropicHeaders(
                request,
                key!,
              ),
            body:
              JSON.stringify(
                anthReq,
              ),
          },
        );

      if (!res.ok) {
        return upstreamErrorResponse(
          res,
          await res.text(),
        );
      }

      if (anthReq.stream) {
        return new Response(
          streamAnthropicToOpenAI(
            res.body as ReadableStream,
            anthReq.model,
          ),
          {
            headers: {
              'Content-Type':
                'text/event-stream',
              'Cache-Control':
                'no-cache',
              Connection:
                'keep-alive',
            },
          },
        );
      }

      const data: any =
        await res.json();

      return new Response(
        JSON.stringify(
          toOpenAIResponse(
            data,
            anthReq.model,
          ),
        ),
        {
          headers: {
            'Content-Type':
              'application/json',
          },
        },
      );
    }

    return fetch(
      `${upstream}/chat/completions`,
      {
        method: 'POST',
        headers:
          openAIHeaders(key!),
        body:
          await request.text(),
      },
    );
  }

  /*
   * ============================================================
   * Model discovery
   * ============================================================
   */
  if (
    route.path === '/v1/models' &&
    request.method === 'GET'
  ) {
    const key =
      extractApiKey(
        request.headers,
      );

    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(err);
    }

    if (
      upstream === GO_UPSTREAM
    ) {
      return fetchGoModels(
        key!,
      );
    }

    if (
      fmt === 'anthropic'
    ) {
      const res =
        await fetch(
          `${upstream}/models`,
          {
            method: 'GET',
            headers:
              anthropicHeaders(
                request,
                key!,
              ),
          },
        );

      if (!res.ok) {
        return upstreamErrorResponse(
          res,
          await res.text(),
        );
      }

      return new Response(
        await res.text(),
        {
          headers: {
            'Content-Type':
              'application/json',
          },
        },
      );
    }

    const res =
      await fetch(
        `${upstream}/models`,
        {
          method: 'GET',
          headers: {
            Authorization:
              `Bearer ${key}`,
          },
        },
      );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    return new Response(
      await res.text(),
      {
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  /*
   * ============================================================
   * Health
   * ============================================================
   */
  return new Response(
    JSON.stringify(
      {
        name:
          'opencode-cowork-proxy',

        upstream,

        routes: {
          '/go':
            GO_UPSTREAM,
          '/zen':
            ZEN_UPSTREAM,
        },

        endpoints: {
          '/v1/messages':
            'Claude / Anthropic Messages compatibility',
          '/v1/chat/completions':
            'OpenAI Chat Completions compatibility',
          '/v1/models':
            'Claude-compatible OpenCode Go model discovery',
        },

        capabilities: {
          modelDiscovery: true,
          tools: true,
          images: true,
          streaming: true,
          reasoningPassthrough: true,
        },
      },
      null,
      2,
    ),
    {
      headers: {
        'Content-Type':
          'application/json',
      },
      status:
        route.path === '/'
          ? 200
          : 404,
    },
  );
}

const app =
  new Hono();

app.all(
  '*',
  (c) =>
    handleRequest(
      c.req.raw,
    ),
);

export default app;