import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  AutoAwesome,
  Refresh,
  ContentCopy,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import type { ReturnsExplanation } from '../../api';

interface DailyReturnsExplanationProps {
  data: ReturnsExplanation | null;
  loading?: boolean;
  onGenerate: () => void;
  onRefresh?: () => void;
}

const DailyReturnsExplanation: React.FC<DailyReturnsExplanationProps> = ({
  data,
  loading = false,
  onGenerate,
  onRefresh,
}) => {
  const theme = useTheme();
  const { t } = useTranslation();
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    if (data?.explanation) {
      navigator.clipboard.writeText(data.explanation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        background: alpha(theme.palette.warning.main, 0.03),
        border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
        borderRadius: 3,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesome sx={{ fontSize: 20, color: 'warning.main' }} />
          <Typography variant="subtitle1" fontWeight={600}>
            {t('portfolio.aiExplanation', 'AI 收益解读')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {data && !loading && (
            <>
              <Tooltip title={copied ? t('common.copied', '已复制') : t('common.copy', '复制')}>
                <IconButton size="small" onClick={handleCopy}>
                  <ContentCopy sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
              {onRefresh && (
                <Tooltip title={t('portfolio.regenerate', '重新生成')}>
                  <IconButton size="small" onClick={onRefresh}>
                    <Refresh sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              )}
            </>
          )}
          {!data && !loading && (
            <Button
              size="small"
              variant="contained"
              color="warning"
              startIcon={<AutoAwesome sx={{ fontSize: 16 }} />}
              onClick={onGenerate}
              sx={{ textTransform: 'none' }}
            >
              {t('portfolio.generateExplanation', '生成解读')}
            </Button>
          )}
        </Box>
      </Box>

      {/* Content */}
      <Paper
        variant="outlined"
        sx={{
          p: 2,
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          borderColor: alpha(theme.palette.warning.main, 0.2),
          borderRadius: 2,
          minHeight: 100,
        }}
      >
        {loading ? (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 80 }}>
            <Box sx={{ textAlign: 'center' }}>
              <CircularProgress size={24} sx={{ mb: 1, color: 'warning.main' }} />
              <Typography variant="body2" color="text.secondary">
                {t('portfolio.generatingExplanation', 'AI 正在分析您的收益情况...')}
              </Typography>
            </Box>
          </Box>
        ) : data ? (
          <>
            <Typography
              variant="body2"
              sx={{
                lineHeight: 1.8,
                whiteSpace: 'pre-wrap',
                color: 'text.primary',
              }}
            >
              {data.explanation}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: 'block', mt: 2, textAlign: 'right' }}
            >
              {t('portfolio.generatedAt', '生成于')} {new Date(data.generated_at).toLocaleString()}
            </Typography>
          </>
        ) : (
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <AutoAwesome sx={{ fontSize: 32, color: alpha(theme.palette.warning.main, 0.4), mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              {t('portfolio.clickToGenerate', '点击"生成解读"让 AI 分析您的今日收益表现')}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
              {t('portfolio.aiExplanationHint', 'AI 将分析主要贡献和拖累因素，并提供简短的操作建议')}
            </Typography>
          </Box>
        )}
      </Paper>
    </Paper>
  );
};

export default DailyReturnsExplanation;
