import {
  extractCachedTokens,
  extractOutputTokens,
  extractUncachedInputTokens,
} from '../../cache';

export function streamOpenAIToAnthropic(
  openaiStream: ReadableStream,
  model: string,
): ReadableStream {
  const messageId = `msg_${Date.now()}`;

  const encoder = new TextEncoder();

  const enqueueSSE = (
    controller: ReadableStreamDefaultController,
    eventType: string,
    data: any,
  ) => {
    controller.enqueue(
      encoder.encode(
        `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`,
      ),
    );
  };

  return new ReadableStream({
    async start(controller) {
      let contentBlockIndex = -1;

      let hasStartedTextBlock = false;
      let hasStartedThinkingBlock = false;
      let isToolUse = false;

      let currentToolCallId: string | null = null;

      const toolCallJsonMap =
        new Map<string, string>();

      let lastUsage: any = null;
      let finishReason: string | null = null;
      let messageStarted = false;

      const reader =
        openaiStream.getReader();

      const decoder =
        new TextDecoder();

      let buffer = '';

      const startMessage = () => {
        if (messageStarted) return;

        enqueueSSE(
          controller,
          'message_start',
          {
            type: 'message_start',
            message: {
              id: messageId,
              type: 'message',
              role: 'assistant',
              content: [],
              model,
              stop_reason: null,
              stop_sequence: null,
              usage: {
                input_tokens: 0,
                output_tokens: 0,
              },
            },
          },
        );

        messageStarted = true;
      };

      const stopCurrentBlock = () => {
        if (
          hasStartedTextBlock ||
          hasStartedThinkingBlock ||
          isToolUse
        ) {
          enqueueSSE(
            controller,
            'content_block_stop',
            {
              type: 'content_block_stop',
              index: contentBlockIndex,
            },
          );
        }

        hasStartedTextBlock = false;
        hasStartedThinkingBlock = false;
        isToolUse = false;
        currentToolCallId = null;
      };

      const getReasoning = (
        delta: any,
      ): string => {
        if (
          typeof delta?.reasoning_content ===
          'string'
        ) {
          return delta.reasoning_content;
        }

        if (
          typeof delta?.reasoning ===
          'string'
        ) {
          return delta.reasoning;
        }

        if (
          Array.isArray(
            delta?.reasoning_details,
          )
        ) {
          return delta.reasoning_details
            .map((item: any) => {
              if (
                typeof item === 'string'
              ) {
                return item;
              }

              if (
                typeof item?.text ===
                'string'
              ) {
                return item.text;
              }

              if (
                typeof item?.content ===
                'string'
              ) {
                return item.content;
              }

              if (
                typeof item?.summary ===
                'string'
              ) {
                return item.summary;
              }

              return '';
            })
            .filter(Boolean)
            .join('');
        }

        if (
          Array.isArray(
            delta?.reasoningDetails,
          )
        ) {
          return delta.reasoningDetails
            .map((item: any) => {
              if (
                typeof item === 'string'
              ) {
                return item;
              }

              if (
                typeof item?.text ===
                'string'
              ) {
                return item.text;
              }

              if (
                typeof item?.content ===
                'string'
              ) {
                return item.content;
              }

              return '';
            })
            .filter(Boolean)
            .join('');
        }

        return '';
      };

      const processStreamDelta = (
        delta: any,
        parsed: any,
      ) => {
        /*
         * Usage
         */
        if (parsed?.usage) {
          lastUsage = {
            input_tokens:
              extractUncachedInputTokens(
                parsed.usage,
              ),

            output_tokens:
              extractOutputTokens(
                parsed.usage,
              ),

            cache_read_input_tokens:
              extractCachedTokens(
                parsed.usage,
              ),

            cache_creation_input_tokens: 0,
          };
        }

        /*
         * Finish reason
         */
        if (
          parsed?.choices?.[0]
            ?.finish_reason
        ) {
          finishReason =
            parsed.choices[0]
              .finish_reason;
        }

        /*
         * Reasoning
         */
        const reasoning =
          getReasoning(delta);

        if (reasoning) {
          if (
            hasStartedTextBlock ||
            isToolUse
          ) {
            stopCurrentBlock();
            contentBlockIndex++;
          }

          if (
            !hasStartedThinkingBlock
          ) {
            if (contentBlockIndex < 0) {
              contentBlockIndex = 0;
            }

            startMessage();

            enqueueSSE(
              controller,
              'content_block_start',
              {
                type:
                  'content_block_start',
                index:
                  contentBlockIndex,
                content_block: {
                  type: 'thinking',
                  thinking: '',
                  signature: '',
                },
              },
            );

            hasStartedThinkingBlock =
              true;
          }

          enqueueSSE(
            controller,
            'content_block_delta',
            {
              type:
                'content_block_delta',
              index:
                contentBlockIndex,
              delta: {
                type:
                  'thinking_delta',
                thinking: reasoning,
              },
            },
          );

          return;
        }

        /*
         * Tool calls
         */
        if (
          Array.isArray(
            delta?.tool_calls,
          ) &&
          delta.tool_calls.length > 0
        ) {
          for (
            const toolCall of
              delta.tool_calls
          ) {
            const toolCallId =
              toolCall.id ||
              currentToolCallId;

            if (!toolCallId) {
              continue;
            }

            if (
              toolCallId !==
              currentToolCallId
            ) {
              if (
                hasStartedTextBlock ||
                hasStartedThinkingBlock ||
                isToolUse
              ) {
                stopCurrentBlock();
                contentBlockIndex++;
              } else if (
                contentBlockIndex < 0
              ) {
                contentBlockIndex = 0;
              }

              isToolUse = true;
              currentToolCallId =
                toolCallId;

              toolCallJsonMap.set(
                toolCallId,
                '',
              );

              startMessage();

              enqueueSSE(
                controller,
                'content_block_start',
                {
                  type:
                    'content_block_start',
                  index:
                    contentBlockIndex,
                  content_block: {
                    type: 'tool_use',
                    id: toolCallId,
                    name:
                      toolCall.function
                        ?.name ||
                      'unknown',
                    input: {},
                  },
                },
              );
            }

            const args =
              toolCall.function
                ?.arguments;

            if (
              typeof args === 'string' &&
              args.length > 0
            ) {
              const previous =
                toolCallJsonMap.get(
                  toolCallId,
                ) || '';

              toolCallJsonMap.set(
                toolCallId,
                previous + args,
              );

              enqueueSSE(
                controller,
                'content_block_delta',
                {
                  type:
                    'content_block_delta',
                  index:
                    contentBlockIndex,
                  delta: {
                    type:
                      'input_json_delta',
                    partial_json:
                      args,
                  },
                },
              );
            }
          }

          return;
        }

        /*
         * Normal text
         */
        if (
          typeof delta?.content ===
            'string' &&
          delta.content.length > 0
        ) {
          if (
            hasStartedThinkingBlock ||
            isToolUse
          ) {
            stopCurrentBlock();
            contentBlockIndex++;
          }

          if (
            !hasStartedTextBlock
          ) {
            if (contentBlockIndex < 0) {
              contentBlockIndex = 0;
            }

            startMessage();

            enqueueSSE(
              controller,
              'content_block_start',
              {
                type:
                  'content_block_start',
                index:
                  contentBlockIndex,
                content_block: {
                  type: 'text',
                  text: '',
                },
              },
            );

            hasStartedTextBlock =
              true;
          }

          enqueueSSE(
            controller,
            'content_block_delta',
            {
              type:
                'content_block_delta',
              index:
                contentBlockIndex,
              delta: {
                type: 'text_delta',
                text: delta.content,
              },
            },
          );
        }
      };

      const processLine = (
        line: string,
      ) => {
        const trimmed = line.trim();

        if (!trimmed) return;

        if (
          !trimmed.startsWith('data:')
        ) {
          return;
        }

        const data =
          trimmed
            .slice(5)
            .trim();

        if (data === '[DONE]') {
          return;
        }

        try {
          const parsed =
            JSON.parse(data);

          const delta =
            parsed?.choices?.[0]
              ?.delta;

          if (delta) {
            processStreamDelta(
              delta,
              parsed,
            );
          }
        } catch {
          /*
           * Ignore incomplete/non-JSON SSE
           * lines.
           */
        }
      };

      try {
        while (true) {
          const {
            done,
            value,
          } = await reader.read();

          if (done) {
            break;
          }

          buffer += decoder.decode(
            value,
            { stream: true },
          );

          const lines =
            buffer.split('\n');

          buffer =
            lines.pop() || '';

          for (
            const line of lines
          ) {
            processLine(line);
          }
        }

        /*
         * Flush decoder.
         */
        buffer += decoder.decode();

        if (buffer.trim()) {
          for (
            const line of
              buffer.split('\n')
          ) {
            processLine(line);
          }
        }
      } finally {
        reader.releaseLock();
      }

      /*
       * Close final content block.
       */
      if (
        hasStartedTextBlock ||
        hasStartedThinkingBlock ||
        isToolUse
      ) {
        stopCurrentBlock();
      }

      if (!messageStarted) {
        startMessage();
      }

      /*
       * Map OpenAI finish reason to Anthropic.
       */
      let stopReason =
        'end_turn';

      if (
        finishReason ===
          'tool_calls' ||
        finishReason ===
          'tool_call'
      ) {
        stopReason = 'tool_use';
      } else if (
        finishReason === 'length'
      ) {
        stopReason = 'max_tokens';
      } else if (
        finishReason ===
        'content_filter'
      ) {
        stopReason =
          'stop_sequence';
      }

      enqueueSSE(
        controller,
        'message_delta',
        {
          type: 'message_delta',
          delta: {
            stop_reason:
              stopReason,
            stop_sequence: null,
          },
          usage:
            lastUsage || {
              input_tokens: 0,
              output_tokens: 0,
            },
        },
      );

      enqueueSSE(
        controller,
        'message_stop',
        {
          type: 'message_stop',
        },
      );

      controller.close();
    },
  });
}