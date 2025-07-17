import { CornerUpRight } from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import './App.css';
import { ChatCompletionMessageParam, MLCEngine } from '@mlc-ai/web-llm';

const MODEL = 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const WebLLMChat: React.FC = () => {
  const [input, setInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);

  const [engine, setEngine] = useState<MLCEngine | null>(null);
  const [status, setStatus] = useState<string>('Инициализация...');

  const [loading, setLoading] = useState<boolean>(false);

  const verifyWebGPU = async (): Promise<boolean> => {
    if (!navigator.gpu) {
      setStatus('WebGPU не поддерживается этим бразуером');
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
      setStatus('Адаптер недостпуен(');
      return false;
    }

    return true;
  };

  const initEngine = useCallback(async (): Promise<void> => {
    if (!(await verifyWebGPU())) return;

    const mlcEngine = new MLCEngine();

    mlcEngine.setInitProgressCallback(report => {
      setStatus(report.text);
    });

    try {
      await mlcEngine.reload(MODEL, {
        temperature: 0.8,
        top_p: 1,
      });

      setEngine(mlcEngine);
      setStatus('Модель готова!');
    } catch (e) {
      setStatus((e as Error).message);
    }
  }, []);

  const sendMessage = async (): Promise<void> => {
    if (!engine || !input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    const assistantMessageIndex = messages.length + 1;

    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: '',
      },
    ]);

    try {
      const stream = await engine.chat.completions.create({
        messages: [...messages, userMessage] as ChatCompletionMessageParam[],
        stream: true,
        stream_options: { include_usage: true },
      });

      let fullContent = '';

      for await (const chunk of stream) {
        const deltaContent = chunk.choices[0]?.delta?.content || '';

        if (deltaContent) {
          fullContent += deltaContent;

          setMessages(prev => {
            const newMessages = [...prev];

            newMessages[assistantMessageIndex] = {
              role: 'assistant',
              content: fullContent,
            };

            return newMessages;
          });
        }

        if (chunk.usage) {
          console.log(chunk.usage);
        }
      }
    } catch (e) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Ошибка: ${(e as Error).message}`,
      };

      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[assistantMessageIndex] = errorMessage;

        return newMessages;
      });
    } finally {
      setLoading(false);
    }
  };

  const handlePressEnter = (e: React.KeyboardEvent): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();

      sendMessage();
    }
  };

  useEffect(() => {
    initEngine();
  }, []);

  return (
    <div className='app-container'>
      <div className='status-container'>
        <div className='status-box'>
          <p className='status-text'>
            Статус: <span className='status-value'>{status}</span>
          </p>
        </div>
      </div>

      <div className='messages-container'>
        <div className='messages-scroll'>
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message-wrapper ${message.role === 'user' ? 'user' : 'assistant'}`}
            >
              <div className='message-bubble'>
                <div className='message-header'>{message.role === 'user' ? 'Вы' : 'Ассистент'}</div>
                {message.content}
              </div>
            </div>
          ))}
        </div>
      </div>

      <footer className='footer'>
        <div className='input-container'>
          <textarea
            className='message-input'
            placeholder='Введите сообщение…'
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={!engine || loading}
            onKeyDown={handlePressEnter}
          />
          <button
            className='send-button'
            disabled={!engine || loading || !input.trim()}
            onClick={sendMessage}
          >
            <CornerUpRight size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
};

export default WebLLMChat;
