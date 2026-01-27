import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Switch,
  FormControlLabel,
  CircularProgress,
  useTheme,
  alpha,
  Collapse,
  IconButton,
  Tooltip,
} from '@mui/material';
import { Chat, ExpandMore, ExpandLess } from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import { create } from 'zustand';
import debounce from 'lodash/debounce';
import ScenarioPresets from './ScenarioPresets';
import type { AIScenario } from './ScenarioPresets';
import StressTestResults from './StressTestResults';
import StressTestChat from './StressTestChat';
import { generateAIScenario } from '../../api';

// Zustand store for stress test state
interface SandboxState {
  scenario: {
    interest_rate_change_bp: number;
    fx_change_pct: number;
    index_change_pct: number;
    oil_change_pct: number;
  };
  selectedPreset: string | null;
  setScenario: (updates: Partial<SandboxState['scenario']>) => void;
  setSelectedPreset: (preset: string | null) => void;
  reset: () => void;
}

const useSandboxStore = create<SandboxState>((set) => ({
  scenario: {
    interest_rate_change_bp: 0,
    fx_change_pct: 0,
    index_change_pct: 0,
    oil_change_pct: 0,
  },
  selectedPreset: null,
  setScenario: (updates) =>
    set((state) => ({
      scenario: { ...state.scenario, ...updates },
    })),
  setSelectedPreset: (preset) => set({ selectedPreset: preset }),
  reset: () =>
    set({
      scenario: {
        interest_rate_change_bp: 0,
        fx_change_pct: 0,
        index_change_pct: 0,
        oil_change_pct: 0,
      },
      selectedPreset: null,
    }),
}));

interface Scenario {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
}

interface StressTestResult {
  projected_pnl: number;
  projected_pnl_pct: number;
  var_95: number;
  var_95_pct: number;
  top_losers: Array<{
    code: string;
    name: string;
    projected_change: number;
    impact_pct: number;
  }>;
  top_gainers: Array<{
    code: string;
    name: string;
    projected_change: number;
    impact_pct: number;
  }>;
  risk_level: string;
}

interface StressTestSandboxProps {
  portfolioId: number;
  scenarios: Scenario[];
  onRunStressTest: (params: {
    scenario_type?: string;
    scenario?: Record<string, number>;
  }) => Promise<StressTestResult>;
  initialResult?: StressTestResult | null;
}

const StressTestSandbox: React.FC<StressTestSandboxProps> = ({
  portfolioId,
  scenarios,
  onRunStressTest,
  initialResult,
}) => {
  const theme = useTheme();
  const { t } = useTranslation();

  const {
    scenario,
    selectedPreset,
    setScenario,
    setSelectedPreset,
  } = useSandboxStore();

  const [result, setResult] = useState<StressTestResult | null>(initialResult || null);
  const [loading, setLoading] = useState(false);

  // AI Scenario State (Phase 1)
  const [aiScenario, setAIScenario] = useState<AIScenario | null>(null);
  const [loadingAICategory, setLoadingAICategory] = useState<string | null>(null);

  // Chat State (Phase 2)
  const [showChat, setShowChat] = useState(false);

  // Debounced stress test execution
  const runStressTestDebounced = useCallback(
    debounce(async (params: { scenario_type?: string; scenario?: Record<string, number> }) => {
      setLoading(true);
      try {
        const response = await onRunStressTest(params);
        setResult(response);
      } catch (error) {
        console.error('Stress test failed:', error);
      } finally {
        setLoading(false);
      }
    }, 500),
    [onRunStressTest]
  );

  // Run stress test when preset changes
  useEffect(() => {
    if (selectedPreset && !selectedPreset.startsWith('ai_')) {
      runStressTestDebounced({ scenario_type: selectedPreset });
    }
  }, [selectedPreset, runStressTestDebounced]);

  const handlePresetSelect = (presetId: string) => {
    if (selectedPreset === presetId) {
      setSelectedPreset(null);
      setResult(null);
    } else {
      setSelectedPreset(presetId);
      // Clear AI scenario when selecting regular preset
      if (!presetId.startsWith('ai_')) {
        setAIScenario(null);
      }
    }
  };

  // AI Scenario Generation (Phase 1)
  const handleRequestAI = async (category: string) => {
    setLoadingAICategory(category);
    try {
      const response = await generateAIScenario(portfolioId, category);
      const newAIScenario: AIScenario = response.scenario;
      setAIScenario(newAIScenario);
    } catch (error) {
      console.error('AI scenario generation failed:', error);
    } finally {
      setLoadingAICategory(null);
    }
  };

  // Handle AI Scenario Selection
  const handleSelectAIScenario = (aiScen: AIScenario) => {
    setSelectedPreset(aiScen.id);

    // Run stress test with AI scenario parameters
    runStressTestDebounced({ scenario: aiScen.parameters });
  };

  // Handle scenario from chat (Phase 2)
  const handleChatScenarioGenerated = (params: Record<string, number>) => {
    // Update sliders with chat-generated params
    setScenario(params as Partial<SandboxState['scenario']>);
  };

  // Handle results from chat
  const handleChatResultsReceived = (chatResult: StressTestResult) => {
    setResult(chatResult);
  };

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        height: '100%',
        background: alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        borderRadius: 3,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="h6" fontWeight={600}>
          {t('portfolio.stressTestSandbox', 'Stress Test Sandbox')}
        </Typography>
        <Tooltip title={t('portfolio.aiChat', 'AI Chat')}>
          <IconButton
            size="small"
            onClick={() => setShowChat(!showChat)}
            sx={{
              color: showChat ? theme.palette.primary.main : theme.palette.text.secondary,
              bgcolor: showChat ? alpha(theme.palette.primary.main, 0.1) : 'transparent',
            }}
          >
            <Chat fontSize="small" />
            {showChat ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        {t('portfolio.stressTestDescription', 'Simulate the impact of macro factor changes on your portfolio')}
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* AI Chat Panel (Phase 2) */}
        <Collapse in={showChat}>
          <StressTestChat
            portfolioId={portfolioId}
            onScenarioGenerated={handleChatScenarioGenerated}
            onResultsReceived={handleChatResultsReceived}
          />
        </Collapse>

        {/* Preset Scenarios with AI Enhancement */}
        <Box>
          <ScenarioPresets
            scenarios={scenarios}
            selectedScenario={selectedPreset}
            onSelect={handlePresetSelect}
            disabled={loading}
            onRequestAI={handleRequestAI}
            loadingAICategory={loadingAICategory}
            aiScenario={aiScenario}
            onSelectAIScenario={handleSelectAIScenario}
          />
        </Box>

        <Divider />

        {/* Results */}
        <Box>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            {t('portfolio.simulationResults', 'Simulation Results')}
          </Typography>

          {loading ? (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                py: 4,
              }}
            >
              <CircularProgress size={32} />
            </Box>
          ) : result ? (
            <StressTestResults
              projectedPnl={result.projected_pnl}
              projectedPnlPct={result.projected_pnl_pct}
              var95={result.var_95}
              var95Pct={result.var_95_pct}
              topLosers={result.top_losers}
              topGainers={result.top_gainers}
              riskLevel={result.risk_level}
            />
          ) : (
            <Box
              sx={{
                py: 4,
                textAlign: 'center',
                color: 'text.secondary',
              }}
            >
              <Typography variant="body2">
                {t('portfolio.selectScenario', 'Select a preset scenario or customize parameters')}
              </Typography>
            </Box>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

export default StressTestSandbox;
