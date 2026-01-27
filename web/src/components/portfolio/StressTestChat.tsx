import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  Chip,
  CircularProgress,
  Fade,
  useTheme,
  alpha,
  Divider,
} from '@mui/material';
import {
  Send,
  AutoAwesome,
  Psychology,
  TrendingDown,
  TrendingUp,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import { stressTestChat } from '../../api';
import type { StressTestChatMessage, StressTestResult } from '../../api';

interface StressTestChatProps {
  portfolioId: number;
  onScenarioGenerated?: (params: Record<string, number>) => void;
  onResultsReceived?: (result: StressTestResult) => void;
}

interface ChatMessage extends StressTestChatMessage {
  timestamp: Date;
  stressResult?: StressTestResult | null;
  scenarioUsed?: Record<string, number> | null;
}

const StressTestChat: React.FC<StressTestChatProps> = ({
  portfolioId,
  onScenarioGenerated,
  onResultsReceived,
}) => {
  const theme = useTheme();
  const { t } = useTranslation();
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const suggestions = [
    t('portfolio.chatSuggestion1', 'What if the Fed raises rates by 75bp?'),
    t('portfolio.chatSuggestion2', 'How would a 3% CNY depreciation affect my portfolio?'),
    t('portfolio.chatSuggestion3', 'Simulate an 8% market crash'),
    t('portfolio.chatSuggestion4', 'What is my biggest risk exposure?'),
  ];

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Build history for API (exclude timestamps and extra data)
      const history: StressTestChatMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await stressTestChat(portfolioId, input.trim(), history);

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        stressResult: response.stress_result,
        scenarioUsed: response.scenario_used,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Notify parent components
      if (response.scenario_used && onScenarioGenerated) {
        onScenarioGenerated(response.scenario_used);
      }
      if (response.stress_result && onResultsReceived) {
        onResultsReceived(response.stress_result);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: t('portfolio.chatError', 'Sorry, I encountered an error. Please try again.'),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderStressResultSummary = (result: StressTestResult) => {
    const isPositive = result.projected_pnl_pct >= 0;
    return (
      <Box
        sx={{
          mt: 1,
          p: 1.5,
          borderRadius: 1,
          bgcolor: alpha(isPositive ? theme.palette.success.main : theme.palette.error.main, 0.1),
          border: `1px solid ${alpha(isPositive ? theme.palette.success.main : theme.palette.error.main, 0.3)}`,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          {isPositive ? (
            <TrendingUp fontSize="small" color="success" />
          ) : (
            <TrendingDown fontSize="small" color="error" />
          )}
          <Typography variant="body2" fontWeight={600}>
            {t('portfolio.projectedPnl', 'Projected P&L')}: {result.projected_pnl_pct >= 0 ? '+' : ''}
            {result.projected_pnl_pct.toFixed(2)}%
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary">
          {t('portfolio.var95', 'VaR (95%)')}: {result.var_95_pct.toFixed(2)}% | {' '}
          {t('portfolio.riskAssessment', 'Risk')}: {result.risk_level}
        </Typography>
      </Box>
    );
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: alpha(theme.palette.background.paper, 0.6),
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          borderBottom: `1px solid ${theme.palette.divider}`,
          background: `linear-gradient(90deg, ${alpha('#FFB300', 0.1)} 0%, transparent 100%)`,
        }}
      >
        <Psychology sx={{ color: '#FFB300' }} />
        <Typography variant="subtitle2" fontWeight={600}>
          {t('portfolio.aiStressTestChat', 'AI Stress Test Assistant')}
        </Typography>
      </Box>

      {/* Messages */}
      <Box
        ref={chatContainerRef}
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
          minHeight: 200,
          maxHeight: 300,
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
            }}
          >
            <AutoAwesome sx={{ fontSize: 40, color: alpha('#FFB300', 0.5) }} />
            <Typography variant="body2" color="text.secondary" textAlign="center">
              {t('portfolio.chatWelcome', 'Ask me about stress scenarios or risk analysis')}
            </Typography>
          </Box>
        ) : (
          messages.map((msg, index) => (
            <Fade in key={index}>
              <Box
                sx={{
                  mb: 2,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                <Box
                  sx={{
                    maxWidth: '85%',
                    p: 1.5,
                    borderRadius: '16px 16px 2px 16px',
                    bgcolor:
                      msg.role === 'user'
                        ? '#ffffff'
                        : alpha(theme.palette.background.default, 0.8),
                    color: msg.role === 'user' ? '#000000' : theme.palette.text.primary,
                    border:
                      msg.role === 'assistant'
                        ? `1px solid ${alpha(theme.palette.divider, 0.2)}`
                        : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    boxShadow: msg.role === 'user' ? '0 2px 8px rgba(0,0,0,0.05)' : 'none',
                  }}
                >
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {msg.content}
                  </Typography>

                  {/* Show stress result summary for assistant messages */}
                  {msg.role === 'assistant' && msg.stressResult && (
                    renderStressResultSummary(msg.stressResult)
                  )}

                  {/* Show scenario parameters if used */}
                  {msg.role === 'assistant' && msg.scenarioUsed && (
                    <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {Object.entries(msg.scenarioUsed).map(([key, value]) => (
                        value !== 0 && (
                          <Chip
                            key={key}
                            size="small"
                            label={`${key.replace(/_/g, ' ')}: ${value > 0 ? '+' : ''}${value}`}
                            sx={{
                              fontSize: '0.65rem',
                              height: 20,
                              bgcolor: alpha(theme.palette.primary.main, 0.1),
                            }}
                          />
                        )
                      ))}
                    </Box>
                  )}
                </Box>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mt: 0.5, px: 0.5 }}
                >
                  {msg.timestamp.toLocaleTimeString()}
                </Typography>
              </Box>
            </Fade>
          ))
        )}
        {isLoading && (
          <Fade in>
            <Box
              sx={{
                mb: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
              }}
            >
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: alpha(theme.palette.background.default, 0.8),
                  border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1
                }}
              >
                <CircularProgress size={16} color="primary" />
                <Typography variant="body2" color="text.secondary">
                  Thinking...
                </Typography>
              </Box>
            </Box>
          </Fade>
        )}
      </Box>

      {/* Suggestions */}
      {messages.length === 0 && (
        <>
          <Divider />
          <Box sx={{ p: 1.5 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              {t('portfolio.trySaying', 'Try saying:')}
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {suggestions.map((suggestion, index) => (
                <Chip
                  key={index}
                  label={suggestion}
                  size="small"
                  onClick={() => handleSuggestionClick(suggestion)}
                  sx={{
                    fontSize: '0.7rem',
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.1),
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        </>
      )}

      {/* Input */}
      <Box
        sx={{
          p: 1.5,
          borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          bgcolor: alpha(theme.palette.background.default, 0.5),
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder={t('portfolio.chatPlaceholder', 'Ask about stress scenarios...')}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: theme.palette.background.paper,
              },
            }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            sx={{
              bgcolor: theme.palette.primary.main,
              color: 'white',
              '&:hover': {
                bgcolor: theme.palette.primary.dark,
              },
              '&.Mui-disabled': {
                bgcolor: alpha(theme.palette.primary.main, 0.3),
                color: alpha('#ffffff', 0.5),
              },
            }}
          >
            {isLoading ? <CircularProgress size={20} color="inherit" /> : <Send />}
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );
};

export default StressTestChat;
