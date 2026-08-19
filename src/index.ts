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

const DEFAULT_UPSTREAM = GO_UPSTREAM;

const API_START_PATHS = new Set(['v1', 'v2']);

type UpstreamProtocol =
  | 'openai'
  | 'anthropic'
  | 'responses';

type ModelRoute = {
  upstreamModel: string;
  protocol: UpstreamProtocol;
  reasoningEfforts?: string[];
};

type RouteConfig = {
  path: string;
  upstream: string;
  modelOverride: string | null;
};

/*
 * Claude Desktop-facing model IDs.
 *
 * IMPORTANT:
 * Claude Desktop validates third-party gateway model names.
 * These therefore use the accepted anthropic/claude-* shape.
 *
 * The actual OpenCode model is in upstreamModel.
 */
const MODEL_ROUTES: Record<string, ModelRoute> = {
  /*
   * Go / OpenAI Chat Completions
   */
  'anthropic/claude-grok-4-5': {
    upstreamModel: 'grok-4.5',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-glm-5-2': {
    upstreamModel: 'glm-5.2',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-glm-5-1': {
    upstreamModel: 'glm-5.1',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-kimi-k3': {
    upstreamModel: 'kimi-k3',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-kimi-k2-7-code': {
    upstreamModel: 'kimi-k2.7-code',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-kimi-k2-6': {
    upstreamModel: 'kimi-k2.6',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-deepseek-v4-pro': {
    upstreamModel: 'deepseek-v4-pro',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-deepseek-v4-flash': {
    upstreamModel: 'deepseek-v4-flash',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-mimo-v2-5-pro': {
    upstreamModel: 'mimo-v2.5-pro',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-mimo-v2-5': {
    upstreamModel: 'mimo-v2.5',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  /*
   * Go / Anthropic Messages
   *
   * OpenCode officially exposes these Go models through /messages.
   */
  'anthropic/claude-minimax-m3': {
    upstreamModel: 'minimax-m3',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-minimax-m2-7': {
    upstreamModel: 'minimax-m2.7',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-qwen3-8-max': {
    upstreamModel: 'qwen3.8-max',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-qwen3-7-max': {
    upstreamModel: 'qwen3.7-max',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-qwen3-7-plus': {
    upstreamModel: 'qwen3.7-plus',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  'anthropic/claude-qwen3-6-plus': {
    upstreamModel: 'qwen3.6-plus',
    protocol: 'anthropic',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  /*
   * Hy3 is Chat Completions.
   */
  'anthropic/claude-hy3': {
    upstreamModel: 'hy3',
    protocol: 'openai',
    reasoningEfforts: [
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ],
  },

  /*
   * GPT 5.6 Luna is included in Go's catalog, but is intentionally
   * not exposed below until the Responses adapter is implemented.
   *
   * Once enabled:
   *
   * anthropic/claude-gpt-5-6-luna -> gpt-5.6-luna -> /responses
   */
};

function stripPrefix(
  path: string,
  prefix: string,
): string | null {
  if (path === prefix) {
    return '/';
  }

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
      path:
        '/' +
        segments.slice(1).join('/'),
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

  return {
    path,
    upstream: DEFAULT_UPSTREAM,
    modelOverride: null,
  };
}

function getUpstream(
  request: Request,
  fallback: string,
): string {
  return (
    request.headers.get(
      'X-Upstream-Url',
    ) || fallback
  );
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

  return (
    MODEL_ROUTES[
      model.trim()
    ] || null
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

function normalizeModelForResponse(
  model: string,
): string {
  return model;
}

async function handleAnthropicMessages(
  request: Request,
  key: string,
  route: RouteConfig,
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
   * ================================================================
   * ANTHROPIC MESSAGES UPSTREAM
   * ================================================================
   */

  if (
    modelRoute.protocol ===
    'anthropic'
  ) {
    const upstream =
      getUpstream(
        request,
        GO_UPSTREAM,
      );

    body.model =
      modelRoute.upstreamModel;

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
            JSON.stringify(
              body,
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
     * Upstream is already Anthropic Messages format.
     *
     * Restore the Claude-facing model ID.
     */
    if (
      !body.stream
    ) {
      const data =
        await res.json();

      data.model =
        normalizeModelForResponse(
          requestedModel,
        );

      return new Response(
        JSON.stringify(
          data,
        ),
        {
          status:
            res.status,
          headers: {
            'Content-Type':
              'application/json',
          },
        },
      );
    }

    /*
     * For Anthropic streaming, we pass the event stream through.
     * The requested model ID remains in the upstream stream's
     * message_start payload in most compatible providers.
     */
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
   * ================================================================
   * OPENAI CHAT UPSTREAM
   * ================================================================
   */

  if (
    modelRoute.protocol ===
    'openai'
  ) {
    const upstream =
      getUpstream(
        request,
        GO_UPSTREAM,
      );

    body.model =
      modelRoute.upstreamModel;

    const openaiRequest =
      formatAnthropicToOpenAI(
        body,
      );

    /*
     * Never lose the real model selected by Claude Desktop.
     */
    openaiRequest.model =
      modelRoute.upstreamModel;

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
   * GPT 5.6 Luna / Responses is intentionally not exposed yet.
   */
  return new Response(
    JSON.stringify({
      type: 'error',
      error: {
        type: 'ModelError',
        message:
          `Model ${requestedModel} uses the OpenAI Responses API. That adapter is not enabled yet.`,
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

function displayName(
  modelId: string,
): string {
  const names: Record<
    string,
    string
  > = {
    'grok-4.5':
      'Grok 4.5',

    'glm-5.2':
      'GLM 5.2',

    'glm-5.1':
      'GLM 5.1',

    'kimi-k3':
      'Kimi K3',

    'kimi-k2.7-code':
      'Kimi K2.7 Code',

    'kimi-k2.6':
      'Kimi K2.6',

    'mimo-v2.5':
      'MiMo V2.5',

    'mimo-v2.5-pro':
      'MiMo V2.5 Pro',

    'minimax-m3':
      'MiniMax M3',

    'minimax-m2.7':
      'MiniMax M2.7',

    'qwen3.8-max':
      'Qwen 3.8 Max',

    'qwen3.7-max':
      'Qwen 3.7 Max',

    'qwen3.7-plus':
      'Qwen 3.7 Plus',

    'qwen3.6-plus':
      'Qwen 3.6 Plus',

    'deepseek-v4-pro':
      'DeepSeek V4 Pro',

    'deepseek-v4-flash':
      'DeepSeek V4 Flash',

    hy3:
      'Hy3',
  };

  return (
    names[modelId] ||
    modelId
  );
}

async function fetchGoCatalog(
  key: string,
): Promise<any[]> {
  const res =
    await fetch(
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

function getUpstreamId(
  model: any,
): string {
  if (
    typeof model?.id !==
    'string'
  ) {
    return '';
  }

  return model.id.replace(
    /^opencode-go\//,
    '',
  );
}

async function fetchModels(
  key: string,
): Promise<Response> {
  /*
   * IMPORTANT:
   *
   * Go only.
   *
   * Never query Zen with the Go subscription credential.
   */
  const catalog =
    await fetchGoCatalog(
      key,
    );

  const available =
    new Set(
      catalog.map(
        getUpstreamId,
      ),
    );

  const models =
    Object.entries(
      MODEL_ROUTES,
    )
      .filter(
        ([, route]) =>
          route.protocol !==
            'responses' &&
          available.has(
            route.upstreamModel,
          ),
      )
      .map(
        ([id, route]) => ({
          id,

          object: 'model',

          created:
            Math.floor(
              Date.now() / 1000,
            ),

          /*
           * Claude Desktop sees an Anthropic provider model.
           * The actual upstream model is exposed only as metadata.
           */
          owned_by:
            'anthropic',

          display_name:
            displayName(
              route.upstreamModel,
            ),

          provider:
            'opencode-go',

          upstream_model:
            route.upstreamModel,

          reasoning_efforts:
            route.reasoningEfforts ||
            [],
        }),
      );

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

async function handleRequest(
  request: Request,
): Promise<Response> {
  const route =
    routeConfig(request);

  if (
    route.path ===
      '/v1/messages' &&
    request.method === 'POST'
  ) {
    const key =
      extractApiKey(
        request.headers,
      );

    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(
        err,
      );
    }

    return handleAnthropicMessages(
      request,
      key!,
      route,
    );
  }

  if (
    route.path ===
      '/v1/models' &&
    request.method === 'GET'
  ) {
    const key =
      extractApiKey(
        request.headers,
      );

    const err =
      validateApiKey(key);

    if (err) {
      return authErrorResponse(
        err,
      );
    }

    return fetchModels(
      key!,
    );
  }

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
      return authErrorResponse(
        err,
      );
    }

    const body =
      await request.text();

    return fetch(
      `${GO_UPSTREAM}/chat/completions`,
      {
        method: 'POST',
        headers:
          openAIHeaders(key!),
        body,
      },
    );
  }

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

          provider:
            'opencode-go',

          modelCount:
            Object.keys(
              MODEL_ROUTES,
            ).length,

          endpoints: {
            messages:
              '/v1/messages',

            models:
              '/v1/models',
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