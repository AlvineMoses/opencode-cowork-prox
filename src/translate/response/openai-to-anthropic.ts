import {
  extractCachedTokens,
  extractOutputTokens,
  extractUncachedInputTokens,
} from '../../cache';

function extractReasoning(
  message: any,
): string {
  if (
    typeof message?.reasoning_content ===
    'string'
  ) {
    return message.reasoning_content;
  }

  if (
    typeof message?.reasoning ===
    'string'
  ) {
    return message.reasoning;
  }

  if (
    typeof message?.reasoning_text ===
    'string'
  ) {
    return message.reasoning_text;
  }

  if (
    Array.isArray(
      message?.reasoning_details,
    )
  ) {
    return message.reasoning_details
      .map(
        (item: any) => {
          if (
            typeof item ===
            'string'
          ) {
            return item;
          }

          return (
            item?.text ||
            item?.content ||
            item?.summary ||
            ''
          );
        },
      )
      .filter(Boolean)
      .join('');
  }

  return '';
}

function extractText(
  message: any,
): string {
  if (
    typeof message?.content ===
    'string'
  ) {
    return message.content;
  }

  if (
    Array.isArray(
      message?.content,
    )
  ) {
    return message.content
      .map(
        (part: any) =>
          part?.type === 'text'
            ? part.text || ''
            : '',
      )
      .filter(Boolean)
      .join('');
  }

  return '';
}

export function formatOpenAIToAnthropic(
  completion: any,
  model: string,
): any {
  const choice =
    completion?.choices?.[0];

  const message =
    choice?.message ||
    {};

  const content: any[] =
    [];

  const reasoning =
    extractReasoning(
      message,
    );

  if (reasoning) {
    content.push({
      type:
        'thinking',

      thinking:
        reasoning,

      signature:
        '',
    });
  }

  const text =
    extractText(
      message,
    );

  if (text) {
    content.push({
      type:
        'text',

      text,
    });
  }

  if (
    Array.isArray(
      message?.tool_calls,
    )
  ) {
    for (
      const toolCall of
        message.tool_calls
    ) {
      let input: any =
        {};

      const raw =
        toolCall?.function
          ?.arguments;

      if (
        typeof raw ===
        'string'
      ) {
        try {
          input =
            JSON.parse(
              raw,
            );
        } catch {
          input = {};
        }
      } else if (
        raw &&
        typeof raw ===
          'object'
      ) {
        input = raw;
      }

      content.push({
        type:
          'tool_use',

        id:
          toolCall?.id ||
          `toolu_${Date.now()}`,

        name:
          toolCall?.function
            ?.name ||
          'unknown',

        input,
      });
    }
  }

  let stopReason =
    'end_turn';

  switch (
    choice?.finish_reason
  ) {
    case 'tool_calls':
    case 'tool_call':
      stopReason =
        'tool_use';
      break;

    case 'length':
      stopReason =
        'max_tokens';
      break;

    case 'content_filter':
      stopReason =
        'stop_sequence';
      break;
  }

  const usage =
    completion?.usage;

  const inputTokens =
    usage
      ? extractUncachedInputTokens(
          usage,
        )
      : 0;

  const outputTokens =
    usage
      ? extractOutputTokens(
          usage,
        )
      : 0;

  const cachedTokens =
    usage
      ? extractCachedTokens(
          usage,
        )
      : 0;

  return {
    id:
      completion?.id ||
      `msg_${Date.now()}`,

    type:
      'message',

    role:
      'assistant',

    content,

    model,

    stop_reason:
      stopReason,

    stop_sequence:
      null,

    usage: {
      input_tokens:
        inputTokens,

      output_tokens:
        outputTokens,

      cache_read_input_tokens:
        cachedTokens,

      cache_creation_input_tokens:
        0,
    },
  };
}