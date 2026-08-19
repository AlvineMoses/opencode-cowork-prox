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
 * OpenCode Go currently exposes models through three API styles:
 *
 * /chat/completions
 * /messages
 * /responses
 *
 * This map is intentionally based on the documented OpenCode Go
 * endpoints rather than assuming every model speaks OpenAI Chat.
 *
 * The catalog itself is fetched dynamically from /models, but these
 * protocol mappings tell the Worker which endpoint to use.
 */
const MODEL_PROTOCOLS: Record<string, UpstreamProtocol> = {
  // OpenAI-compatible Chat Completions
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

  // Anthropic Messages
  'minimax-m3': 'anthropic',
  'minimax-m2.7': 'anthropic',
  'minimax-m2.5': 'anthropic',

  'qwen3.8-max': 'anthropic',
  'qwen3.7-max': 'anthropic',
  'qwen3.7-plus': 'anthropic',
  'qwen3.6-plus': 'anthropic',

  // OpenAI Responses
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

function normalizeModelId(model: unknown): string | null {
  if (typeof model !== 'string') {
    return null;
  }

  if (model.startsWith('opencode-go/')) {
    return model.slice('opencode-go/'.length);
  }

  return model;
}

function getModelProtocol(
  model: string | null,
): UpstreamProtocol {
  if (!model) {
    return 'openai';
  }

  const normalized = normalizeModelId(model);

  if (!normalized) {
    return 'openai';
  }

  return MODEL_PROTOCOLS[normalized] || 'openai';
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
    const value = res.headers.get(name);

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
 * Claude Desktop sends Anthropic Messages requests.
 *
 * For OpenAI-compatible models:
 *
 * Claude
 *   -> Anthropic Messages
 *   -> OpenAI Chat translation
 *   -> OpenCode /chat/completions
 *
 * For Anthropic-compatible Go models:
 *
 * Claude
 *   -> Anthropic Messages
 *   -> OpenCode /messages
 *
 * For Responses models:
 *
 * Claude
 *   -> Anthropic Messages
 *   -> Responses adapter
 *
 * The latter requires a dedicated Responses translator if you want
 * GPT 5.6 Luna to work natively. Until that adapter exists, we return
 * a clear error instead of silently sending an incompatible request.
 */
async function handleAnthropicMessages(
  request: Request,
  route: RouteConfig,
  upstream: string,
  key: string,
): Promise<Response> {
  const req = await request.json();

  const selectedModel =
    route.modelOverride ||
    normalizeModelId(req.model);

  if (route.modelOverride) {
    req.model = route.modelOverride;
  } else if (selectedModel) {
    req.model = selectedModel;
  }

  const protocol =
    getModelProtocol(selectedModel);

  /*
   * IMPORTANT:
   *
   * Do not replace the selected model just because an image is present.
   *
   * Claude Desktop can attach images while using a model selected from
   * the model picker. The upstream model's actual vision capability
   * should determine whether it is usable.
   */
  const requestHasImages = hasImages(req);

  if (protocol === 'openai') {
    const openaiReq =
      formatAnthropicToOpenAI(req);

    const res = await fetch(
      `${upstream}/chat/completions`,
      {
        method: 'POST',
        headers: openAIHeaders(key),
        body: JSON.stringify(openaiReq),
      },
    );

    if (!res.ok) {
      return upstreamErrorResponse(
        res,
        await res.text(),
      );
    }

    const originalModel =
      req.model;

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

  if (protocol === 'anthropic') {
    /*
     * MiniMax/Qwen Go models expose Anthropic's Messages API.
     *
     * Claude Desktop is already speaking Anthropic Messages, so this
     * path can preserve tool calls, thinking blocks and image content
     * without unnecessarily converting the request to OpenAI format.
     */

    const res = await fetch(
      `${upstream}/messages`,
      {
        method: 'POST',
        headers: anthropicHeaders(
          request,
          key,
        ),
        body: JSON.stringify(req),
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

  /*
   * GPT 5.6 Luna uses OpenAI Responses.
   *
   * We deliberately don't fake a Chat -> Responses conversion here.
   * The request/response semantics are different enough that doing so
   * would risk breaking tools, reasoning and streaming.
   *
   * Add a dedicated Responses adapter before exposing this model as
   * fully supported.
   */
  if (protocol === 'responses') {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            `Model ${selectedModel} uses the OpenAI Responses API and is not yet enabled by this compatibility layer.`,
          model: selectedModel,
          requestHasImages,
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
          `Unsupported upstream protocol for model ${selectedModel}.`,
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

/**
 * Fetch the current OpenCode Go model catalog.
 *
 * We don't hard-code the model list into Claude Desktop.
 * OpenCode explicitly exposes the current Go catalog at /models.
 */
async function fetchGoModels(
  key: string,
): Promise<Response> {
  const res = await fetch(
    `${GO_UPSTREAM}/models`,
    {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${key}`,
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

  /*
   * Claude Desktop only needs the normal model objects.
   *
   * We normalize the IDs so Claude sees:
   *
   *   kimi-k3
   *
   * rather than:
   *
   *   opencode-go/kimi-k3
   *
   * The Worker can accept either form later.
   */
  const models =
    Array.isArray(body?.data)
      ? body.data.map(
          (model: ModelInfo) => ({
            ...model,
            id:
              normalizeModelId(
                model.id,
              ) || model.id,
            object:
              model.object ||
              'model',
          }),
        )
      : [];

  return new Response(
    JSON.stringify({
      object: 'list',
      data: models,
    }),
    {
      status: res.status,
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
   * Anthropic → upstream
   *
   * This is the path Claude Desktop/Cowork uses.
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
     * The /go route always means OpenCode Go.
     *
     * If X-Upstream-Format is explicitly set to anthropic, preserve
     * the old generic behavior. Otherwise route according to the
     * selected Go model.
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
     * Generic Anthropic upstream pass-through.
     */
    if (fmt === 'anthropic') {
      const body =
        await request.text();

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
            body,
          },
        );

      return res;
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
   * OpenAI → Anthropic
   */
  if (
    route.path === '/v1/chat/completions' &&
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

    if (fmt === 'anthropic') {
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
            await request.text(),
        },
      );

    return res;
  }

  /*
   * Model discovery
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

    /*
     * For /go, always expose the current Go catalog.
     */
    if (
      upstream === GO_UPSTREAM
    ) {
      return fetchGoModels(
        key!,
      );
    }

    /*
     * Generic Anthropic upstream.
     */
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

    /*
     * Generic OpenAI upstream.
     */
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
   * Health / root endpoint
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
            'Anthropic Messages → OpenCode Go',
          '/v1/chat/completions':
            'OpenAI Chat Completions compatibility',
          '/v1/models':
            'Dynamic OpenCode Go model discovery',
        },

        capabilities: {
          modelDiscovery: true,
          tools: true,
          images: true,
          streaming: true,
          reasoningPassthrough: true,
        },

        note:
          'GPT 5.6 Luna requires a dedicated Responses adapter before it can be used through the Anthropic compatibility route.',
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