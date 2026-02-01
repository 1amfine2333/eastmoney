import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  Button,
  Chip,
  IconButton,
  Collapse
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import { fetchStockAIDiagnosis } from '../../api';
import type { DiagnosisResponse } from '../../api';

interface AIDiagnosisCardProps {
  code: string;
  stockName: string;
  onClose?: () => void;
}

export default function AIDiagnosisCard({ code, stockName, onClose }: AIDiagnosisCardProps) {
  const { t } = useTranslation();
  const [data, setData] = useState<DiagnosisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDiagnosis = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchStockAIDiagnosis(code);
      setData(result);
    } catch (err) {
      setError(t('stocks.ai.diagnosis_error'));
      console.error('Failed to load AI diagnosis:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDiagnosis();
  }, [code]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#22c55e';
    if (score >= 60) return '#3b82f6';
    if (score >= 40) return '#f59e0b';
    return '#ef4444';
  };

  const getRecommendationColor = (rec: string) => {
    if (rec.includes('买入')) return '#22c55e';
    if (rec.includes('持有')) return '#f59e0b';
    if (rec.includes('减持') || rec.includes('卖出')) return '#ef4444';
    return '#64748b';
  };

  const renderScoreGauge = (score: number) => {
    const color = getScoreColor(score);
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference * (1 - score / 100);

    return (
      <Box sx={{ position: 'relative', width: 120, height: 120, mx: 'auto' }}>
        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="8"
          />
          {/* Score arc */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 0.5s ease-in-out' }}
          />
        </svg>
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center'
        }}>
          <Typography variant="h4" sx={{ fontWeight: 900, color, fontFamily: 'JetBrains Mono' }}>
            {score}
          </Typography>
          <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 700 }}>
            /100
          </Typography>
        </Box>
      </Box>
    );
  };

  if (loading) {
    return (
      <Paper elevation={0} sx={{
        p: 4,
        borderRadius: '16px',
        border: '1px solid #e2e8f0',
        bgcolor: '#fcfcfc',
        textAlign: 'center'
      }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 4 }}>
          <Box sx={{ position: 'relative' }}>
            <CircularProgress size={48} sx={{ color: '#6366f1' }} />
            <AutoAwesomeIcon sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: '#6366f1',
              fontSize: 20
            }} />
          </Box>
          <Typography sx={{ color: '#64748b', fontWeight: 600 }}>
            {t('stocks.ai.analyzing')}
          </Typography>
          <Typography variant="caption" sx={{ color: '#94a3b8' }}>
            {t('stocks.ai.analyzing_hint')}
          </Typography>
        </Box>
      </Paper>
    );
  }

  if (error || !data) {
    return (
      <Paper elevation={0} sx={{
        p: 4,
        borderRadius: '16px',
        border: '1px solid #fecaca',
        bgcolor: '#fef2f2',
        textAlign: 'center'
      }}>
        <Typography color="error" sx={{ fontWeight: 600 }}>{error || t('stocks.ai.no_data')}</Typography>
        <Button
          startIcon={<RefreshIcon />}
          onClick={loadDiagnosis}
          sx={{ mt: 2, color: '#ef4444' }}
        >
          {t('common.refresh')}
        </Button>
      </Paper>
    );
  }

  if (!data.diagnosis) {
    return (
      <Paper elevation={0} sx={{
        p: 4,
        borderRadius: '16px',
        border: '1px solid #fecaca',
        bgcolor: '#fef2f2',
        textAlign: 'center'
      }}>
        <Typography color="error" sx={{ fontWeight: 600 }}>
          {t('stocks.ai.no_data')}
        </Typography>
        <Button
          startIcon={<RefreshIcon />}
          onClick={loadDiagnosis}
          sx={{ mt: 2, color: '#ef4444' }}
        >
          {t('common.refresh')}
        </Button>
      </Paper>
    );
  }

  const { diagnosis } = data;

  return (
    <Paper elevation={0} sx={{
      borderRadius: '16px',
      border: '1px solid #e2e8f0',
      overflow: 'hidden',
      bgcolor: '#ffffff'
    }}>
      {/* Header */}
      <Box sx={{
        p: 2,
        bgcolor: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <AutoAwesomeIcon sx={{ color: '#fff', fontSize: 24 }} />
          <Box>
            <Typography variant="subtitle1" sx={{ color: '#fff', fontWeight: 800 }}>
              {t('stocks.ai.diagnosis_title')}
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.8)' }}>
              {stockName}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton size="small" onClick={loadDiagnosis} sx={{ color: '#fff' }}>
            <RefreshIcon fontSize="small" />
          </IconButton>
          {onClose && (
            <IconButton size="small" onClick={onClose} sx={{ color: '#fff' }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      </Box>

      {/* Score Section */}
      <Box sx={{ p: 3, bgcolor: '#f8fafc', borderBottom: '1px solid #f1f5f9' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ flex: 1 }}>
            {renderScoreGauge(diagnosis.score)}
          </Box>
          <Box sx={{ flex: 1, textAlign: 'center' }}>
            <Typography variant="overline" sx={{ color: '#64748b', fontWeight: 800 }}>
              {t('stocks.ai.rating')}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 900, color: getScoreColor(diagnosis.score) }}>
              {diagnosis.rating}
            </Typography>
            <Chip
              label={diagnosis.recommendation}
              size="small"
              sx={{
                mt: 1,
                bgcolor: getRecommendationColor(diagnosis.recommendation) + '20',
                color: getRecommendationColor(diagnosis.recommendation),
                fontWeight: 800,
                fontSize: '0.75rem'
              }}
            />
          </Box>
        </Box>
      </Box>

      {/* Highlights */}
      <Box sx={{ p: 2.5, borderBottom: '1px solid #f1f5f9' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <TrendingUpIcon sx={{ color: '#22c55e', fontSize: 20 }} />
          <Typography variant="overline" sx={{ color: '#22c55e', fontWeight: 900 }}>
            {t('stocks.ai.highlights')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {diagnosis.highlights.map((highlight, idx) => (
            <Box key={idx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Box sx={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                bgcolor: '#22c55e',
                mt: 0.8,
                flexShrink: 0
              }} />
              <Typography variant="body2" sx={{ color: '#334155', fontSize: '0.85rem' }}>
                {highlight}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      {/* Risks */}
      <Box sx={{ p: 2.5, borderBottom: '1px solid #f1f5f9' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <WarningAmberIcon sx={{ color: '#f59e0b', fontSize: 20 }} />
          <Typography variant="overline" sx={{ color: '#f59e0b', fontWeight: 900 }}>
            {t('stocks.ai.risks')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {diagnosis.risks.map((risk, idx) => (
            <Box key={idx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Box sx={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                bgcolor: '#f59e0b',
                mt: 0.8,
                flexShrink: 0
              }} />
              <Typography variant="body2" sx={{ color: '#334155', fontSize: '0.85rem' }}>
                {risk}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      {/* Action Advice */}
      <Box sx={{ p: 2.5, bgcolor: '#f0fdf4', borderBottom: '1px solid #dcfce7' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <LightbulbIcon sx={{ color: '#16a34a', fontSize: 20 }} />
          <Typography variant="overline" sx={{ color: '#16a34a', fontWeight: 900 }}>
            {t('stocks.ai.action_advice')}
          </Typography>
        </Box>
        <Typography variant="body2" sx={{ color: '#166534', lineHeight: 1.6 }}>
          {diagnosis.action_advice}
        </Typography>
      </Box>

      {/* Key Focus */}
      <Box sx={{ p: 2, bgcolor: '#fef3c7' }}>
        <Typography variant="caption" sx={{ color: '#92400e', fontWeight: 700 }}>
          {t('stocks.ai.key_focus')}: {diagnosis.key_focus}
        </Typography>
      </Box>

      {/* Timestamp */}
      <Box sx={{ px: 2, py: 1, bgcolor: '#f8fafc', textAlign: 'right' }}>
        <Typography variant="caption" sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>
          {t('stocks.ai.generated_at')}: {new Date(data.data_timestamp).toLocaleString()}
        </Typography>
      </Box>
    </Paper>
  );
}
