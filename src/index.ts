import { Hono } from 'hono';
import {
  extractApiKey,
  validateApiKey,
  authErrorResponse,
} from './auth';

import { formatAnthropicToOpenAI } from './translate/request/anthropic-to-openai';
import { formatOpenAIToAnthropic } from './translate/request/openai-to-anthropic';

import {
  formatOpenAIToAnthropic as toAnthropicResponse,
} from './translate/response/openai-to-anthropic';

import {
  formatAnthropicToOpenAI as toOpenAIResponse,
} from './translate/response/anthropic-to-openai';

import { streamOpenAIToAnthropic } from './translate/stream/openai-to-anthropic';
import { streamAnthropicToOpenAI } from './translate/stream/anthropic-to-openai';

const GO_UPSTREAM = 'https://opencode.ai/zen/go/v1';
const ZEN_UPSTREAM = 'https://opencode.ai/zen/v1';
const DEFAULT_UPSTREAM = GO_UPSTREAM;

const API_START_PATHS = new Set(['v1', 'v2']);

type UpstreamProtocol =
  | 'openai'
  | 'anthropic'
  | 'responses';

type UpstreamSource = 'go' | 'zen';

type RouteConfig = {
  path: string;
  upstream: string;
  modelOverride: string | null;
};

type ModelRoute = {
  upstreamModel: string;
  protocol: UpstreamProtocol;
  source: UpstreamSource;
  familyTier?: 'sonnet' | 'opus' | 'haiku';
  reasoningEfforts?: string[];
};

/*
 * Claude Desktop requires Anthropic-looking model IDs.
 *
 * The left side is what Claude Desktop sees.
 * The right side is what OpenCode receives.
 *
 * Keep these IDs stable; they are the contract between Claude Desktop
 * and this Worker.
 */
const MODEL_ROUTES: Record<string, ModelRoute> = {
  /*
   * Go
   */
  'anthropic/claude-glm-5-2': {
    upstreamModel: 'glm-5.2',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-glm-5-1': {
    upstreamModel: 'glm-5.1',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-grok-4-5': {
    upstreamModel: 'grok-4.5',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-kimi-k3': {
    upstreamModel: 'kimi-k3',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-kimi-k2-7-code': {
    upstreamModel: 'kimi-k2.7-code',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-kimi-k2-6': {
    upstreamModel: 'kimi-k2.6',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-deepseek-v4-pro': {
    upstreamModel: 'deepseek-v4-pro',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-deepseek-v4-flash': {
    upstreamModel: 'deepseek-v4-flash',
    protocol: 'openai',
    source: 'go',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-mimo-v2-5-pro': {
    upstreamModel: 'mimo-v2.5-pro',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-mimo-v2-5': {
    upstreamModel: 'mimo-v2.5',
    protocol: 'openai',
    source: 'go',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-minimax-m3': {
    upstreamModel: 'minimax-m3',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-minimax-m2-7': {
    upstreamModel: 'minimax-m2.7',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-minimax-m2-5': {
    upstreamModel: 'minimax-m2.5',
    protocol: 'openai',
    source: 'go',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-qwen3-8-max': {
    upstreamModel: 'qwen3.8-max',
    protocol: 'anthropic',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-qwen3-7-max': {
    upstreamModel: 'qwen3.7-max',
    protocol: 'anthropic',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-qwen3-7-plus': {
    upstreamModel: 'qwen3.7-plus',
    protocol: 'anthropic',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-qwen3-6-plus': {
    upstreamModel: 'qwen3.6-plus',
    protocol: 'anthropic',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-hy3': {
    upstreamModel: 'hy3',
    protocol: 'openai',
    source: 'go',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  /*
   * Zen
   */
  'anthropic/claude-glm-5': {
    upstreamModel: 'glm-5',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-kimi-k2-5': {
    upstreamModel: 'kimi-k2.5',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-grok-build-0-1': {
    upstreamModel: 'grok-build-0.1',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'sonnet',
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },

  'anthropic/claude-big-pickle': {
    upstreamModel: 'big-pickle',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high'],
  },

  'anthropic/claude-mimo-v2-5-free': {
    upstreamModel: 'mimo-v2.5-free',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high'],
  },

  'anthropic/claude-deepseek-v4-flash-free': {
    upstreamModel: 'deepseek-v4-flash-free',
    protocol: 'openai',
    source: 'zen',
    familyTier: 'haiku',
    reasoningEfforts: ['low', 'medium', 'high'],
  },

  /*
   * These Zen models already speak Anthropic Messages.
   */
  'claude-sonnet-4-5': {
    upstreamModel: 'claude-sonnet-4-5',
    protocol: 'anthropic',
    source: 'zen',
    familyTier: 'sonnet',
    reasoningEfforts: ['high', 'max'],
  },

  'claude-haiku-4-5': {
    upstreamModel: 'claude-haiku-4-5',
    protocol: 'anthropic',
    source: 'zen',
    familyTier: 'haiku',
    reasoningEfforts: ['high', 'max'],
  },

  /*
   * GPT Responses models are intentionally NOT added here yet.
   *
   * They need a Responses <-> Anthropic adapter before they should
   * appear in Claude's picker.
   */
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

function extractModelSegment(
  path: string,
): {
  path: string;
  model: string | null;
} {
  const segments = path
    .replace(/^\/+/, '')
    .split('/');

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

function routeConfig(
  request: Request,
): RouteConfig {
  const path =
    new URL(request.url).pathname;

  const goPath =
    stripPrefix(path, '/go');

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

  const zenPath =
    stripPrefix(path, '/zen');

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

  return {
    path,
    upstream: DEFAULT_UPSTREAM,
    modelOverride: null,
  };
}

function getUpstreamBase(
  source: UpstreamSource,
): string {
  return source === 'zen'
    ? ZEN_UPSTREAM
    : GO_UPSTREAM;
}

function getUpstream(
  request: Request,
  routeUpstream: string,
): string {
  return (
    request.headers.get(
      'X-Upstream-Url',
    ) ||
    routeUpstream
  );
}

function upstreamFormat(
  request: Request,
): 'openai' | 'anthropic' {
  const fmt = (
    request.headers.get(
      'X-Upstream-Format',
    ) || 'openai'
  ).toLowerCase();

  return fmt === 'anthropic'
    ? 'anthropic'
    : 'openai';
}

function anthropicHeaders(
  request: Request,
  key: string,
): Record<string, string> {
  const headers: Record<
    string,
    string
  > = {
    'Content-Type':
      'application/json',

    'X-Api-Key': key,

    'Anthropic-Version':
      request.headers.get(
        'Anthropic-Version',
      ) ||
      '2023-06-01',
  };

  const beta =
    request.headers.get(
      'Anthropic-Beta',
    );

  if (beta) {
    headers['Anthropic-Beta'] =
      beta;
  }

  return headers;
}

function openAIHeaders(
  key: string,
): Record<string, string> {
  return {
    'Content-Type':
      'application/json',
    Authorization:
      `Bearer ${key}`,
  };
}

function getModelRoute(
  model: unknown,
): ModelRoute | null {
  if (
    typeof model !== 'string'
  ) {
    return null;
  }

  const id =
    model.trim();

  return (
    MODEL_ROUTES[id] ||
    null
  );
}

function upstreamErrorResponse(
  res: Response,
  body: string,
): Response {
  const headers =
    new Headers();

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
      headers.set(
        name,
        value,
      );
    }
  }

  return new Response(
    body,
    {
      status: res.status,
      headers,
    },
  );
}

/*
 * ============================================================================
 * ANTHROPIC -> OPENAI request routing
 * ============================================================================
 */

async function handleAnthropicMessages(
  request: Request,
  route: RouteConfig,
  key: string,
): Promise<Response> {
  const body =
    await request.json();

  const requestedModel =
    route.modelOverride ||
    body?.model;

  const modelRoute =
    getModelRoute(
      requestedModel,
    );

  if (!modelRoute) {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            `Model ${requestedModel} is not configured in the gateway.`,
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
   * Send the REAL OpenCode model upstream.
   */
  body.model =
    modelRoute.upstreamModel;

  /*
   * --------------------------------------------------------------------------
   * Models that already speak Anthropic Messages
   * --------------------------------------------------------------------------
   */

  if (
    modelRoute.protocol ===
    'anthropic'
  ) {
    const upstream =
      getUpstreamBase(
        modelRoute.source,
      );

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
            JSON.stringify(body),
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
        status:
          res.status,
        headers:
          res.headers,
      },
    );
  }

  /*
   * --------------------------------------------------------------------------
   * OpenAI-compatible models
   * --------------------------------------------------------------------------
   */

  if (
    modelRoute.protocol ===
    'openai'
  ) {
    const openaiRequest =
      formatAnthropicToOpenAI(
        body,
      );

    /*
     * Preserve the real OpenCode model.
     */
    openaiRequest.model =
      modelRoute.upstreamModel;

    const upstream =
      getUpstreamBase(
        modelRoute.source,
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
              openaiRequest,
            ),
        },
      );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    /*
     * Claude Desktop expects its selected model ID back.
     */
    if (
      openaiRequest.stream
    ) {
      return new Response(
        streamOpenAIToAnthropic(
          res.body as ReadableStream,
          requestedModel,
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

    const data =
      await res.json();

    return new Response(
      JSON.stringify(
        toAnthropicResponse(
          data,
          requestedModel,
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
   * Responses is deliberately not silently converted.
   */
  return new Response(
    JSON.stringify({
      type: 'error',
      error: {
        type: 'ModelError',
        message:
          `Model ${requestedModel} uses the Responses API, which is not yet supported by this bridge.`,
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

/*
 * ============================================================================
 * MODEL DISCOVERY
 * ============================================================================
 */

function openCodeApiModel(
  model: any,
): string {
  return typeof model?.id === 'string'
    ? model.id.replace(
        /^opencode-go\//,
        '',
      )
    : '';
}

async function fetchCatalog(
  source: UpstreamSource,
  key: string,
): Promise<any[]> {
  const upstream =
    getUpstreamBase(
      source,
    );

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
    return [];
  }

  const body =
    await res.json();

  return Array.isArray(
    body?.data,
  )
    ? body.data
    : [];
}

function routeExistsInCatalog(
  modelId: string,
  sourceModels: any[],
): boolean {
  return sourceModels.some(
    model =>
      openCodeApiModel(model) ===
      modelId,
  );
}

async function fetchGoAndZenModels(
  key: string,
): Promise<Response> {
  const [
    goModels,
    zenModels,
  ] = await Promise.all([
    fetchCatalog(
      'go',
      key,
    ),
    fetchCatalog(
      'zen',
      key,
    ),
  ]);

  const result: any[] = [];

  for (
    const [
      id,
      route,
    ] of Object.entries(
      MODEL_ROUTES,
    )
  ) {
    const sourceModels =
      route.source === 'go'
        ? goModels
        : zenModels;

    if (
      !routeExistsInCatalog(
        route.upstreamModel,
        sourceModels,
      )
    ) {
      continue;
    }

    result.push({
      id,
      object: 'model',
      created:
        Math.floor(
          Date.now() / 1000,
        ),
      owned_by:
        'anthropic',
      display_name:
        modelDisplayName(
          route.upstreamModel,
        ),
      provider:
        route.source === 'go'
          ? 'opencode-go'
          : 'opencode-zen',
      upstream_model:
        route.upstreamModel,
      reasoning_efforts:
        route.reasoningEfforts ||
        [],
    });
  }

  /*
   * Keep the gateway probe model.
   *
   * This maps to GLM 5.2 only for authentication / initial probing.
   */
  result.unshift({
    id:
      'claude-sonnet-4-5',
    object: 'model',
    created:
      Math.floor(
        Date.now() / 1000,
      ),
    owned_by:
      'anthropic',
    display_name:
      'GLM 5.2',
    provider:
      'opencode-go',
    upstream_model:
      'glm-5.2',
    reasoning_efforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  });

  return new Response(
    JSON.stringify({
      object: 'list',
      data: dedupeModels(
        result,
      ),
    }),
    {
      headers: {
        'Content-Type':
          'application/json',
      },
    },
  );
}

function modelDisplayName(
  modelId: string,
): string {
  const names: Record<
    string,
    string
  > = {
    'glm-5.2': 'GLM 5.2',
    'glm-5.1': 'GLM 5.1',
    'glm-5': 'GLM 5',
    'grok-4.5': 'Grok 4.5',
    'grok-build-0.1':
      'Grok Build 0.1',
    'kimi-k3': 'Kimi K3',
    'kimi-k2.7-code':
      'Kimi K2.7 Code',
    'kimi-k2.6':
      'Kimi K2.6',
    'kimi-k2.5':
      'Kimi K2.5',
    'deepseek-v4-pro':
      'DeepSeek V4 Pro',
    'deepseek-v4-flash':
      'DeepSeek V4 Flash',
    'deepseek-v4-flash-free':
      'DeepSeek V4 Flash Free',
    'mimo-v2.5-pro':
      'MiMo V2.5 Pro',
    'mimo-v2.5':
      'MiMo V2.5',
    'mimo-v2.5-free':
      'MiMo V2.5 Free',
    'minimax-m3':
      'MiniMax M3',
    'minimax-m2.7':
      'MiniMax M2.7',
    'minimax-m2.5':
      'MiniMax M2.5',
    'qwen3.8-max':
      'Qwen 3.8 Max',
    'qwen3.7-max':
      'Qwen 3.7 Max',
    'qwen3.7-plus':
      'Qwen 3.7 Plus',
    'qwen3.6-plus':
      'Qwen 3.6 Plus',
    'hy3': 'Hy3',
    'big-pickle':
      'Big Pickle',
    'claude-sonnet-4-5':
      'Claude Sonnet 4.5',
    'claude-haiku-4-5':
      'Claude Haiku 4.5',
  };

  return (
    names[modelId] ||
    modelId
  );
}

function dedupeModels(
  models: any[],
): any[] {
  const seen =
    new Set<string>();

  return models.filter(
    model => {
      if (
        seen.has(model.id)
      ) {
        return false;
      }

      seen.add(model.id);
      return true;
    },
  );
}

/*
 * ============================================================================
 * MAIN REQUEST HANDLER
 * ============================================================================
 */

async function handleRequest(
  request: Request,
): Promise<Response> {
  const route =
    routeConfig(request);

  const key =
    extractApiKey(
      request.headers,
    );

  /*
   * Anthropic Messages
   */
  if (
    route.path ===
      '/v1/messages' &&
    request.method === 'POST'
  ) {
    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(
        err,
      );
    }

    /*
     * Explicit route URL override still works.
     */
    if (
      route.modelOverride
    ) {
      return handleAnthropicMessages(
        request,
        route,
        key!,
      );
    }

    /*
     * Normal root gateway.
     */
    return handleAnthropicMessages(
      request,
      route,
      key!,
    );
  }

  /*
   * OpenAI -> Anthropic compatibility.
   */
  if (
    route.path ===
      '/v1/chat/completions' &&
    request.method === 'POST'
  ) {
    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(
        err,
      );
    }

    const fmt =
      upstreamFormat(
        request,
      );

    if (
      fmt === 'anthropic'
    ) {
      const req =
        await request.json();

      const anthReq =
        formatOpenAIToAnthropic(
          req,
        );

      const upstream =
        getUpstream(
          request,
          ZEN_UPSTREAM,
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

      if (
        anthReq.stream
      ) {
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

      const data =
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

    const res =
      await fetch(
        `${DEFAULT_UPSTREAM}/chat/completions`,
        {
          method: 'POST',
          headers:
            openAIHeaders(
              key!,
            ),
          body:
            await request.text(),
        },
      );

    return res;
  }

  /*
   * Model discovery.
   */
  if (
    route.path ===
      '/v1/models' &&
    request.method === 'GET'
  ) {
    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(
        err,
      );
    }

    return fetchGoAndZenModels(
      key!,
    );
  }

  /*
   * Health endpoint.
   */
  if (
    route.path === '/'
  ) {
    return new Response(
      JSON.stringify(
        {
          name:
            'opencode-cowork-proxy',
          status:
            'ok',
          models:
            Object.keys(
              MODEL_ROUTES,
            ).length,
          sources: [
            'opencode-go',
            'opencode-zen',
          ],
        },
        null,
        2,
      ),
      {
        headers: {
          'Content-Type':
            'application/json',
        },
      },
    );
  }

  return new Response(
    JSON.stringify({
      error:
        'Not found',
    }),
    {
      status: 404,
      headers: {
        'Content-Type':
          'application/json',
      },
    },
  );
}

const app =
  new Hono();

app.all(
  '*',
  c =>
    handleRequest(
      c.req.raw,
    ),
);

export default app;