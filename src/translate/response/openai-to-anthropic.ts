/**
 * Converts OpenAI Chat Completions response to Anthropic Messages response.
 *
 * Important:
 * - Preserves reasoning as Anthropic `thinking` content blocks.
 * - Supports reasoning_content, reasoning, and common reasoning_details shapes.
 * - Preserves normal text and tool calls.
 * - Preserves usage/cache token information.
 */

import {
  extractCachedTokens,
  extractOutputTokens,
  extractUncachedInputTokens,
} from '../../cache';

function extractReasoning(message: any): string {
  if (!message) return '';

  // OpenAI-compatible providers commonly use reasoning_content.
  if (typeof message.reasoning_content === 'string') {
    return message.reasoning_content;
  }

  // Some providers use `reasoning`.
  if (typeof message.reasoning === 'string') {
    return message.reasoning;
  }

  // Some providers return reasoning as an array.
  if (Array.isArray(message.reasoning)) {
    return message.reasoning
      .map((item: any) => {
        if (typeof item === 'string') return item;

        if (typeof item?.text === 'string') {
          return item.text;
        }

        if (typeof item?.content === 'string') {
          return item.content;
        }

        return '';
      })
      .filter(Boolean)
      .join('');
  }

  // OpenAI-style reasoning_details / reasoningDetails.
  const details =
    message.reasoning_details ??
    message.reasoningDetails;

  if (Array.isArray(details)) {
    return details
      .map((item: any) => {
        if (typeof item === 'string') return item;

        if (typeof item?.text === 'string') {
          return item.text;
        }

        if (typeof item?.content === 'string') {
          return item.content;
        }

        if (typeof item?.summary === 'string') {
          return item.summary;
        }

        return '';
      })
      .filter(Boolean)
      .join('');
  }

  return '';
}

function extractText(message: any): string {
  if (!message) return '';

  if (typeof message.content === 'string') {
    return message.content;
  }

  if (Array.isArray(message.content)) {
    return message.content
      .map((part: any) => {
        if (typeof part === 'string') {
          return part;
        }

        if (part?.type === 'text') {
          return typeof part.text === 'string'
            ? part.text
            : '';
        }

        return '';
      })
      .filter(Boolean)
      .join('');
  }

  return '';
}

export function formatOpenAIToAnthropic(
  completion: any,
  model: string,
): any {
  const choice = completion?.choices?.[0];
  const message = choice?.message || {};

  const content: any[] = [];

  /*
   * IMPORTANT:
   *
   * Anthropic expects thinking to be a content block:
   *
   * {
   *   type: "thinking",
   *   thinking: "..."
   * }
   *
   * Do this BEFORE text so reasoning appears before the answer.
   */
  const reasoning = extractReasoning(message);

  if (reasoning) {
    content.push({
      type: 'thinking',
      thinking: reasoning,
      signature: '',
    });
  }

  const text = extractText(message);

  if (text) {
    content.push({
      type: 'text',
      text,
    });
  }

  /*
   * Convert OpenAI tool calls into Anthropic tool_use blocks.
   */
  if (Array.isArray(message.tool_calls)) {
    for (const toolCall of message.tool_calls) {
      if (
        !toolCall ||
        toolCall.type !== 'function'
      ) {
        continue;
      }

      let input: any = {};

      const rawArguments =
        toolCall.function?.arguments;

      if (typeof rawArguments === 'string') {
        try {
          input = JSON.parse(rawArguments);
        } catch {
          /*
           * Keep malformed arguments usable rather than crashing
           * the entire response.
           */
          input = {};
        }
      } else if (
        rawArguments &&
        typeof rawArguments === 'object'
      ) {
        input = rawArguments;
      }

      content.push({
        type: 'tool_use',
        id:
          toolCall.id ||
          `toolu_${Date.now()}`,
        name:
          toolCall.function?.name ||
          'unknown',
        input,
      });
    }
  }

  /*
   * Always return a valid Anthropic message.
   */
  let stopReason = 'end_turn';

  if (
    choice?.finish_reason === 'tool_calls' ||
    choice?.finish_reason === 'tool_call'
  ) {
    stopReason = 'tool_use';
  } else if (
    choice?.finish_reason === 'length'
  ) {
    stopReason = 'max_tokens';
  } else if (
    choice?.finish_reason === 'content_filter'
  ) {
    stopReason = 'stop_sequence';
  } else if (
    typeof choice?.finish_reason === 'string' &&
    choice.finish_reason
  ) {
    stopReason = 'end_turn';
  }

  const usage = completion?.usage;

  const inputTokens = usage
    ? extractUncachedInputTokens(usage)
    : 0;

  const outputTokens = usage
    ? extractOutputTokens(usage)
    : 0;

  const cacheReadTokens = usage
    ? extractCachedTokens(usage)
    : 0;

  return {
    id:
      completion?.id ||
      `msg_${Date.now()}`,

    type: 'message',

    role: 'assistant',

    content,

    model:
      completion?.model ||
      model,

    stop_reason: stopReason,

    stop_sequence: null,

    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cache_read_input_tokens:
        cacheReadTokens,
      cache_creation_input_tokens: 0,
    },
  };
}