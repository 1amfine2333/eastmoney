import React from 'react';
import {
  Box,
  Button,
  Typography,
  useTheme,
  alpha,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CurrencyYuan,
  ShowChart,
  Memory,
  LocalGasStation,
  AutoAwesome,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';

interface Scenario {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
}

export interface AIScenario {
  id: string;
  name: string;
  parameters: {
    interest_rate_change_bp: number;
    fx_change_pct: number;
    index_change_pct: number;
    oil_change_pct: number;
  };
  reasoning: string;
  confidence: 'high' | 'medium' | 'low';
  generated_at: string;
  source: 'ai' | 'fallback';
}

interface ScenarioPresetsProps {
  scenarios: Scenario[];
  selectedScenario: string | null;
  onSelect: (scenarioId: string) => void;
  disabled?: boolean;
  onRequestAI?: (category: string) => void;
  loadingAICategory?: string | null;
  aiScenario?: AIScenario | null;
  onSelectAIScenario?: (scenario: AIScenario) => void;
}

const ScenarioPresets: React.FC<ScenarioPresetsProps> = ({
  scenarios,
  selectedScenario,
  onSelect,
  disabled = false,
  onRequestAI,
  loadingAICategory,
  aiScenario,
  onSelectAIScenario,
}) => {
  const theme = useTheme();
  const { t } = useTranslation();

  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'trending_up':
        return <TrendingUp fontSize="small" />;
      case 'trending_down':
        return <TrendingDown fontSize="small" />;
      case 'currency_yuan':
        return <CurrencyYuan fontSize="small" />;
      case 'show_chart':
        return <ShowChart fontSize="small" />;
      case 'memory':
        return <Memory fontSize="small" />;
      case 'local_gas_station':
        return <LocalGasStation fontSize="small" />;
      default:
        return <ShowChart fontSize="small" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'monetary_policy':
        return theme.palette.primary.main;
      case 'currency':
        return theme.palette.warning.main;
      case 'market':
        return theme.palette.error.main;
      case 'sector':
        return theme.palette.info.main;
      case 'commodity':
        return theme.palette.success.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'monetary_policy':
        return t('portfolio.monetaryPolicy', 'Monetary Policy');
      case 'currency':
        return t('portfolio.currency', 'Currency');
      case 'market':
        return t('portfolio.market', 'Market');
      case 'sector':
        return t('portfolio.sector', 'Sector');
      case 'commodity':
        return t('portfolio.commodity', 'Commodity');
      default:
        return category;
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return theme.palette.success.main;
      case 'medium':
        return theme.palette.warning.main;
      case 'low':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  // Group scenarios by category
  const groupedScenarios = scenarios.reduce((acc, scenario) => {
    if (!acc[scenario.category]) {
      acc[scenario.category] = [];
    }
    acc[scenario.category].push(scenario);
    return acc;
  }, {} as Record<string, Scenario[]>);

  const isAIScenarioSelected = aiScenario && selectedScenario === aiScenario.id;

  return (
    <Box>
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        {t('portfolio.presetScenarios', 'Preset Scenarios')}
      </Typography>

      {Object.entries(groupedScenarios).map(([category, categoryScenarios]) => (
        <Box key={category} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Chip
              size="small"
              label={getCategoryLabel(category)}
              sx={{
                bgcolor: alpha(getCategoryColor(category), 0.1),
                color: getCategoryColor(category),
                fontSize: '0.7rem',
              }}
            />
            {onRequestAI && (
              <Tooltip title={t('portfolio.aiSuggest', 'AI Suggestion')}>
                <IconButton
                  size="small"
                  onClick={() => onRequestAI(category)}
                  disabled={disabled || loadingAICategory === category}
                  sx={{
                    color: '#FFB300',
                    '&:hover': {
                      bgcolor: alpha('#FFB300', 0.1),
                    },
                  }}
                >
                  {loadingAICategory === category ? (
                    <CircularProgress size={16} sx={{ color: '#FFB300' }} />
                  ) : (
                    <AutoAwesome fontSize="small" />
                  )}
                </IconButton>
              </Tooltip>
            )}
          </Box>

          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {categoryScenarios.map((scenario) => (
              <Button
                key={scenario.id}
                variant={selectedScenario === scenario.id ? 'contained' : 'outlined'}
                size="small"
                startIcon={getIcon(scenario.icon)}
                onClick={() => onSelect(scenario.id)}
                disabled={disabled}
                sx={{
                  borderRadius: 2,
                  textTransform: 'none',
                  borderColor: selectedScenario === scenario.id
                    ? getCategoryColor(scenario.category)
                    : alpha(theme.palette.divider, 0.5),
                  bgcolor: selectedScenario === scenario.id
                    ? getCategoryColor(scenario.category)
                    : 'transparent',
                  color: selectedScenario === scenario.id
                    ? 'white'
                    : theme.palette.text.primary,
                  '&:hover': {
                    borderColor: getCategoryColor(scenario.category),
                    bgcolor: selectedScenario === scenario.id
                      ? getCategoryColor(scenario.category)
                      : alpha(getCategoryColor(scenario.category), 0.08),
                  },
                }}
              >
                {scenario.name}
              </Button>
            ))}

            {/* Show AI Scenario if available for this category */}
            {aiScenario && aiScenario.id.includes(category) && (
              <Tooltip
                title={
                  <Box>
                    <Typography variant="caption" fontWeight={600}>
                      {t('portfolio.aiReasoning', 'AI Reasoning')}:
                    </Typography>
                    <Typography variant="caption" display="block">
                      {aiScenario.reasoning}
                    </Typography>
                    <Box sx={{ mt: 0.5 }}>
                      <Chip
                        size="small"
                        label={`${t('portfolio.confidence', 'Confidence')}: ${aiScenario.confidence}`}
                        sx={{
                          fontSize: '0.65rem',
                          height: 18,
                          bgcolor: alpha(getConfidenceColor(aiScenario.confidence), 0.2),
                          color: getConfidenceColor(aiScenario.confidence),
                        }}
                      />
                    </Box>
                  </Box>
                }
                arrow
              >
                <Button
                  variant={isAIScenarioSelected ? 'contained' : 'outlined'}
                  size="small"
                  startIcon={<AutoAwesome fontSize="small" />}
                  onClick={() => onSelectAIScenario?.(aiScenario)}
                  disabled={disabled}
                  sx={{
                    borderRadius: 2,
                    textTransform: 'none',
                    borderColor: isAIScenarioSelected
                      ? '#FFB300'
                      : '#FFB300',
                    borderWidth: 2,
                    bgcolor: isAIScenarioSelected
                      ? '#FFB300'
                      : 'transparent',
                    color: isAIScenarioSelected
                      ? 'white'
                      : '#FFB300',
                    '&:hover': {
                      borderColor: '#FFA000',
                      bgcolor: isAIScenarioSelected
                        ? '#FFA000'
                        : alpha('#FFB300', 0.1),
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {aiScenario.name}
                    <Chip
                      size="small"
                      label="AI"
                      sx={{
                        fontSize: '0.6rem',
                        height: 16,
                        bgcolor: isAIScenarioSelected ? alpha('#ffffff', 0.3) : alpha('#FFB300', 0.2),
                        color: isAIScenarioSelected ? 'white' : '#FFB300',
                        ml: 0.5,
                      }}
                    />
                  </Box>
                </Button>
              </Tooltip>
            )}
          </Box>
        </Box>
      ))}
    </Box>
  );
};

export default ScenarioPresets;
