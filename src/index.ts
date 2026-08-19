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

type ModelRoute = {
  upstreamModel: string;
  protocol: UpstreamProtocol;
};

/*
 * ============================================================================
 * MODEL ROUTING
 * ============================================================================
 *
 * Claude Desktop expects models exposed by a gateway to look like Anthropic
 * models. We therefore expose Anthropic-shaped IDs while internally routing
 * them to the actual OpenCode model.
 *
 * Example:
 *
 *   Claude Desktop
 *        |
 *        | model = claude-sonnet-4-5
 *        v
 *   Our Worker
 *        |
 *        | model = glm-5.2
 *        v
 *   OpenCode Go
 *
 * This is the critical compatibility layer.
 */
const MODEL_ROUTES: Record<string, ModelRoute> = {
  /*
   * Claude Desktop's gateway credential probe.
   *
   * Claude probes this model even though the actual inference is performed
   * by OpenCode's glm-5.2 model.
   */
  'claude-sonnet-4-5': {
    upstreamModel: 'glm-5.2',
    protocol: 'openai',
  },

  /*
   * OpenCode models exposed through Anthropic-looking IDs.
   *
   * These IDs are intentionally prefixed with:
   *
   *   anthropic/claude-opencode-
   *
   * because Claude Desktop validates the model name before allowing it.
   */

  'anthropic/claude-opencode-minimax-m3': {
    upstreamModel: 'minimax-m3',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-minimax-m2-7': {
    upstreamModel: 'minimax-m2.7',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-minimax-m2-5': {
    upstreamModel: 'minimax-m2.5',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-kimi-k3': {
    upstreamModel: 'kimi-k3',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-kimi-k2-7-code': {
    upstreamModel: 'kimi-k2.7-code',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-kimi-k2-6': {
    upstreamModel: 'kimi-k2.6',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-glm-5-2': {
    upstreamModel: 'glm-5.2',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-glm-5-1': {
    upstreamModel: 'glm-5.1',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-deepseek-v4-pro': {
    upstreamModel: 'deepseek-v4-pro',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-deepseek-v4-flash': {
    upstreamModel: 'deepseek-v4-flash',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-qwen3-7-max': {
    upstreamModel: 'qwen3.7-max',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-qwen3-8-max': {
    upstreamModel: 'qwen3.8-max',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-qwen3-7-plus': {
    upstreamModel: 'qwen3.7-plus',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-qwen3-6-plus': {
    upstreamModel: 'qwen3.6-plus',
    protocol: 'anthropic',
  },

  'anthropic/claude-opencode-mimo-v2-5-pro': {
    upstreamModel: 'mimo-v2.5-pro',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-mimo-v2-5': {
    upstreamModel: 'mimo-v2.5',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-hy3': {
    upstreamModel: 'hy3',
    protocol: 'openai',
  },

  'anthropic/claude-opencode-grok-4-5': {
    upstreamModel: 'grok-4.5',
    protocol: 'openai',
  },

  /*
   * GPT-5.6 Luna intentionally isn't exposed yet.
   *
   * It uses OpenAI Responses rather than Chat Completions, and this Worker
   * does not currently contain the Responses <-> Anthropic adapter.
   *
   * DO NOT advertise it to Claude until that adapter exists.
   */
};

/*
 * ============================================================================
 * PATH ROUTING
 * ============================================================================
 */

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

/*
 * ============================================================================
 * UPSTREAM HELPERS
 * ============================================================================
 */

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

/*
 * ============================================================================
 * MODEL HELPERS
 * ============================================================================
 */

function normalizeModelId(
  model: unknown,
): string | null {
  if (typeof model !== 'string') {
    return null;
  }

  const value = model.trim();

  if (!value) {
    return null;
  }

  return value;
}

function getModelRoute(
  model: string | null,
): ModelRoute | null {
  if (!model) {
    return null;
  }

  return (
    MODEL_ROUTES[model] ||
    null
  );
}

/*
 * ============================================================================
 * REQUEST HELPERS
 * ============================================================================
 */

function hasImages(
  body: any,
): boolean {
  const messages =
    body?.messages;

  if (!Array.isArray(messages)) {
    return false;
  }

  return messages.some(
    (msg: any) =>
      Array.isArray(
        msg.content,
      ) &&
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
 * ANTHROPIC MESSAGES HANDLER
 * ============================================================================
 *
 * Claude Desktop/Cowork speaks Anthropic Messages.
 *
 * Depending on the selected model, we translate that request into the
 * appropriate OpenCode protocol.
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
   * Preserve the model Claude actually requested.
   *
   * This is important because the response sent back to Claude should
   * continue identifying the Claude-facing model rather than exposing
   * the internal OpenCode model.
   */
  const requestedModel =
    route.modelOverride ||
    normalizeModelId(
      req.model,
    );

  if (!requestedModel) {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            'No model was specified.',
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
            `Model ${requestedModel} is not supported`,
          model:
            requestedModel,
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
   * THIS IS THE KEY FIX.
   *
   * Claude sends:
   *
   *   claude-sonnet-4-5
   *
   * We send OpenCode:
   *
   *   glm-5.2
   *
   * The Claude-facing model ID is retained separately as requestedModel.
   */
  req.model =
    modelRoute.upstreamModel;

  const protocol =
    modelRoute.protocol;

  const requestHasImages =
    hasImages(req);

  /*
   * --------------------------------------------------------------------------
   * OPENAI CHAT COMPLETIONS
   * --------------------------------------------------------------------------
   */

  if (
    protocol === 'openai'
  ) {
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

    /*
     * Always return the Claude-facing model ID.
     */
    const responseModel =
      requestedModel;

    if (
      openaiReq.stream
    ) {
      return new Response(
        streamOpenAIToAnthropic(
          res.body as ReadableStream,
          responseModel,
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
          responseModel,
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
   * --------------------------------------------------------------------------
   * ANTHROPIC MESSAGES
   * --------------------------------------------------------------------------
   *
   * These models already speak Anthropic Messages.
   *
   * No conversion is necessary.
   */

  if (
    protocol === 'anthropic'
  ) {
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
        status:
          res.status,

        headers:
          res.headers,
      },
    );
  }

  /*
   * --------------------------------------------------------------------------
   * RESPONSES API
   * --------------------------------------------------------------------------
   *
   * Currently intentionally disabled.
   */

  if (
    protocol === 'responses'
  ) {
    return new Response(
      JSON.stringify({
        type: 'error',
        error: {
          type: 'ModelError',
          message:
            `Model ${requestedModel} uses the OpenAI Responses API and is not yet enabled by this compatibility layer.`,
          model:
            requestedModel,
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
          `Unsupported upstream protocol for model ${requestedModel}.`,
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
 * ============================================================================
 * MODEL DISCOVERY
 * ============================================================================
 *
 * Claude Desktop uses /v1/models while validating the gateway.
 *
 * We fetch OpenCode's real catalog, then translate it into the model IDs
 * Claude expects.
 */

async function fetchGoModels(
  key: string,
): Promise<Response> {
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
    return upstreamErrorResponse(
      res,
      await res.text(),
    );
  }

  const body: any =
    await res.json();

  const upstreamModels =
    Array.isArray(
      body?.data,
    )
      ? body.data
      : [];

  const models: ModelInfo[] =
    [];

  /*
   * Build Claude-facing models from our explicit route table.
   *
   * This prevents OpenCode's internal model IDs from leaking into Claude.
   */
  for (
    const [
      claudeModelId,
      route,
    ] of Object.entries(
      MODEL_ROUTES,
    )
  ) {
    /*
     * Don't advertise Responses models until supported.
     */
    if (
      route.protocol ===
      'responses'
    ) {
      continue;
    }

    /*
     * Only expose a model if OpenCode actually reports it.
     *
     * The exception is claude-sonnet-4-5 below, because it is our
     * compatibility alias for glm-5.2.
     */
    const upstreamModel =
      upstreamModels.find(
        (model: any) => {
          const id =
            typeof model?.id ===
            'string'
              ? model.id.replace(
                  /^opencode-go\//,
                  '',
                )
              : '';

          return (
            id ===
            route.upstreamModel
          );
        },
      );

    if (
      !upstreamModel &&
      claudeModelId !==
        'claude-sonnet-4-5'
    ) {
      continue;
    }

    models.push({
      ...(upstreamModel || {}),

      id:
        claudeModelId,

      object:
        'model',

      created:
        upstreamModel?.created ||
        Math.floor(
          Date.now() / 1000,
        ),

      owned_by:
        'anthropic',

      openCodeModel:
        route.upstreamModel,

      display_name:
        claudeModelId ===
        'claude-sonnet-4-5'
          ? 'Claude Sonnet 4.5 (OpenCode)'
          : `OpenCode — ${route.upstreamModel}`,

      provider:
        'opencode-go',
    });
  }

  /*
   * Ensure Claude's gateway probe model ALWAYS exists.
   *
   * Claude probes this exact ID during Setup.
   */
  if (
    !models.some(
      model =>
        model.id ===
        'claude-sonnet-4-5',
    )
  ) {
    models.unshift({
      id:
        'claude-sonnet-4-5',

      object:
        'model',

      created:
        Math.floor(
          Date.now() / 1000,
        ),

      owned_by:
        'anthropic',

      openCodeModel:
        'glm-5.2',

      display_name:
        'Claude Sonnet 4.5 (OpenCode)',

      provider:
        'opencode-go',
    });
  }

  return new Response(
    JSON.stringify({
      object: 'list',
      data: models,
    }),
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
 * ============================================================================
 * MAIN REQUEST HANDLER
 * ============================================================================
 */

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
   * --------------------------------------------------------------------------
   * ANTHROPIC -> UPSTREAM
   * --------------------------------------------------------------------------
   *
   * Claude Desktop/Cowork uses this endpoint.
   */

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

    /*
     * OpenCode Go is our normal gateway route.
     *
     * Route individual models based on MODEL_ROUTES.
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
     * Explicit Anthropic upstream.
     */
    if (
      fmt === 'anthropic'
    ) {
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

    if (
      route.modelOverride
    ) {
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

    if (
      openaiReq.stream
    ) {
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
   * --------------------------------------------------------------------------
   * OPENAI -> ANTHROPIC
   * --------------------------------------------------------------------------
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
      return authErrorResponse(
        err,
      );
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
   * --------------------------------------------------------------------------
   * MODEL DISCOVERY
   * --------------------------------------------------------------------------
   */

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

    /*
     * OpenCode Go model discovery.
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
   * --------------------------------------------------------------------------
   * HEALTH / ROOT
   * --------------------------------------------------------------------------
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
            'Anthropic Messages → OpenCode model routing',

          '/v1/chat/completions':
            'OpenAI Chat Completions compatibility',

          '/v1/models':
            'Claude-compatible OpenCode model discovery',
        },

        capabilities: {
          modelDiscovery:
            true,

          tools:
            true,

          images:
            true,

          streaming:
            true,

          reasoningPassthrough:
            true,
        },

        modelRouting:
          Object.fromEntries(
            Object.entries(
              MODEL_ROUTES,
            ).map(
              ([
                id,
                route,
              ]) => [
                id,
                route.upstreamModel,
              ],
            ),
          ),

        note:
          'Claude-facing model IDs are mapped internally to OpenCode models. GPT-5.6 Luna is not advertised until a Responses API adapter is implemented.',
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

/*
 * ============================================================================
 * HONO
 * ============================================================================
 */

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