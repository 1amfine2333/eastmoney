import { useState, useEffect, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Typography, 
  Button, 
  Paper, 
  CircularProgress, 
  Alert,
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListSubheader,
  Tooltip,
  IconButton
} from '@mui/material';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import HistoryIcon from '@mui/icons-material/History';
import ArticleIcon from '@mui/icons-material/Article';
import DownloadIcon from '@mui/icons-material/Download';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import DeleteIcon from '@mui/icons-material/Delete';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import html2canvas from 'html2canvas';
import { runSentimentAnalysis, fetchSentimentReports, fetchReportContent, deleteSentimentReport } from '../api';
import type { SentimentReportItem } from '../api';

export default function SentimentPage() {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<SentimentReportItem[]>([]);
  const [exporting, setExporting] = useState<boolean>(false);

  // Ref for export
  const reportRef = useRef<HTMLDivElement>(null);

  const loadHistory = async () => {
    try {
      const data = await fetchSentimentReports();
      setHistory(data);
      // Auto-load latest if none selected
      if (data.length > 0 && !selectedFile) {
          handleSelectReport(data[0].filename);
      }
    } catch (err) {
      console.error("Failed to load history", err);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  // Group history by Date (YYYY-MM-DD)
  const groupedHistory = useMemo(() => {
    const groups: Record<string, SentimentReportItem[]> = {};
    history.forEach(item => {
        // Assume date format is somewhat standard, extract YYYY-MM-DD
        // The API returns 'date' which might be full datetime string. 
        // We'll parse it.
        try {
            const dateObj = new Date(item.date);
            const dateKey = isNaN(dateObj.getTime()) ? 'Unknown Date' : dateObj.toLocaleDateString();
            if (!groups[dateKey]) groups[dateKey] = [];
            groups[dateKey].push(item);
        } catch (e) {
             if (!groups['Others']) groups['Others'] = [];
             groups['Others'].push(item);
        }
    });
    // Sort keys descending (newest dates first)
    return Object.entries(groups).sort((a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime());
  }, [history]);

  const handleRunAnalysis = async () => {
    setAnalyzing(true);
    setError(null);
    try {
      const data = await runSentimentAnalysis();
      setReport(data.report);
      setSelectedFile(data.filename);
      await loadHistory(); 
    } catch (err) {
      console.error(err);
      setError(t('sentiment.error_gen'));
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSelectReport = async (filename: string) => {
    setLoading(true);
    setError(null);
    setSelectedFile(filename);
    try {
      const content = await fetchReportContent(filename);
      setReport(content);
    } catch (err) {
      console.error(err);
      setError(t('sentiment.error_load'));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteReport = async (e: React.MouseEvent, filename: string) => {
      e.stopPropagation();
      if (!window.confirm('Are you sure you want to delete this report?')) return;

      try {
          await deleteSentimentReport(filename);
          // If deleted the current one, clear it
          if (selectedFile === filename) {
              setReport(null);
              setSelectedFile(null);
          }
          await loadHistory();
      } catch (err) {
          console.error("Failed to delete report", err);
          setError('Failed to delete report');
      }
  };

  // Export report as image
  const handleExportImage = async () => {
      if (!reportRef.current || !report) return;
      setExporting(true);
      try {
          const exportWidth = 1200;
          const originalWidth = reportRef.current.style.width;
          const originalMaxWidth = reportRef.current.style.maxWidth;
          
          reportRef.current.style.width = `${exportWidth}px`;
          reportRef.current.style.maxWidth = `${exportWidth}px`;
          
          const canvas = await html2canvas(reportRef.current, {
              scale: 2,
              useCORS: true,
              backgroundColor: '#ffffff',
              logging: false,
              width: exportWidth,
              windowWidth: exportWidth
          });
          
          reportRef.current.style.width = originalWidth;
          reportRef.current.style.maxWidth = originalMaxWidth;
          
          const link = document.createElement('a');
          const fileName = `sentiment_${selectedFile || 'report'}.png`;
          link.download = fileName;
          link.href = canvas.toDataURL('image/png');
          link.click();
      } catch (error) {
          console.error('Failed to export image:', error);
      } finally {
          setExporting(false);
      }
  };

  return (
    <div className="h-full flex flex-col bg-slate-50">
      {/* Minimalist Header */}
      <div className="px-6 py-3 bg-white border-b border-slate-200 flex justify-between items-center shrink-0">
        <div className="flex items-center gap-3">
            <PsychologyIcon sx={{ color: '#6366f1', fontSize: 28 }} />
            <div>
                <Typography variant="h6" className="font-bold text-slate-800 tracking-tight" sx={{ lineHeight: 1.2 }}>
                    {t('sentiment.title')}
                </Typography>
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
                    <Typography variant="caption" className="text-slate-400 font-medium text-[10px] uppercase tracking-wider">
                        {t('sentiment.active')}
                    </Typography>
                </div>
            </div>
        </div>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
                variant="outlined"
                size="small"
                startIcon={analyzing ? <CircularProgress size={14} color="inherit" /> : <AutoGraphIcon fontSize="small" />}
                onClick={handleRunAnalysis}
                disabled={analyzing}
                sx={{
                    color: '#0f172a',
                    borderColor: '#e2e8f0',
                    borderRadius: '20px',
                    px: 2,
                    textTransform: 'none',
                    fontWeight: 600,
                    fontSize: '0.8rem',
                    '&:hover': { bgcolor: '#f8fafc', borderColor: '#cbd5e1' }
                }}
            >
                {analyzing ? t('reports.processing') : t('sentiment.run_analysis')}
            </Button>
        </Box>
      </div>

      <div className="flex-1 overflow-hidden flex">
        {/* Sidebar History - Grouped by Date */}
        <div className="w-72 bg-white border-r border-slate-200 overflow-y-auto flex flex-col shrink-0 custom-scrollbar">
          <div className="p-4 border-b border-slate-100 bg-slate-50 sticky top-0 z-10">
             <div className="flex items-center text-slate-500">
                <HistoryIcon fontSize="small" className="mr-2"/>
                <Typography variant="subtitle2" fontWeight="700" sx={{ letterSpacing: '0.05em' }}>{t('sentiment.timeline')}</Typography>
             </div>
          </div>
          
          <List disablePadding sx={{ pb: 4 }}>
            {groupedHistory.map(([dateLabel, items]) => (
                <li key={dateLabel}>
                    <ul className="p-0 list-none">
                        <ListSubheader sx={{ 
                            bgcolor: '#fff', 
                            color: '#94a3b8', 
                            fontWeight: 800, 
                            fontSize: '0.75rem', 
                            lineHeight: '40px',
                            fontFamily: 'JetBrains Mono'
                        }}>
                            {dateLabel === new Date().toLocaleDateString() ? t('sentiment.today') : dateLabel}
                        </ListSubheader>
                        {items.map((item) => {
                            // Extract time from date string if possible
                            const timeStr = new Date(item.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                            return (
                                <ListItem key={item.filename} disablePadding>
                                    <ListItemButton 
                                        selected={selectedFile === item.filename}
                                        onClick={() => handleSelectReport(item.filename)}
                                        className="group"
                                        sx={{
                                            mx: 1.5,
                                            my: 0.5,
                                            borderRadius: '8px',
                                            border: '1px solid transparent',
                                            '&.Mui-selected': {
                                                bgcolor: '#eff6ff',
                                                borderColor: '#dbeafe',
                                                color: '#2563eb',
                                                '&:hover': { bgcolor: '#dbeafe' }
                                            },
                                            '&:hover': { bgcolor: '#f8fafc' }
                                        }}
                                    >
                                        <ListItemText 
                                            primary={
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <AccessTimeIcon sx={{ fontSize: 14, color: selectedFile === item.filename ? '#2563eb' : '#94a3b8' }} />
                                                    <Typography sx={{ fontWeight: 700, fontSize: '0.85rem' }}>{timeStr}</Typography>
                                                </Box>
                                            }
                                            secondary={item.filename.replace('sentiment_', '').replace('.md', '')}
                                            secondaryTypographyProps={{ 
                                                fontSize: '0.7rem', 
                                                noWrap: true, 
                                                sx: { mt: 0.5, color: selectedFile === item.filename ? '#60a5fa' : '#cbd5e1' } 
                                            }}
                                        />
                                        <IconButton 
                                            size="small" 
                                            onClick={(e) => handleDeleteReport(e, item.filename)}
                                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                                            sx={{ 
                                                color: '#94a3b8', 
                                                '&:hover': { color: '#ef4444', bgcolor: 'rgba(239, 68, 68, 0.1)' },
                                                padding: '4px',
                                                mr: -1
                                            }}
                                        >
                                            <DeleteIcon style={{ fontSize: 16 }} />
                                        </IconButton>
                                    </ListItemButton>
                                </ListItem>
                            );
                        })}
                    </ul>
                </li>
            ))}
            
            {history.length === 0 && (
              <div className="p-8 text-center">
                <Typography sx={{ color: '#cbd5e1', fontSize: '0.8rem', fontWeight: 600 }}>{t('sentiment.no_signals')}</Typography>
              </div>
            )}
          </List>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-8 bg-slate-50/50">
          {error && (
            <Alert severity="error" sx={{ mb: 3, borderRadius: '8px' }}>
              {error}
            </Alert>
          )}
          
          {/* Export Bar */}
          {report && !loading && (
            <div className="flex justify-end mb-4">
                <Tooltip title={t('reports.download_image')}>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={exporting ? <CircularProgress size={16} /> : <DownloadIcon />}
                        onClick={handleExportImage}
                        disabled={exporting}
                        className="border-slate-300 text-slate-600 hover:bg-slate-50 hover:border-slate-400"
                    >
                        {exporting ? t('reports.processing') : t('reports.export')}
                    </Button>
                </Tooltip>
            </div>
          )}

          <Box sx={{ position: 'relative', minHeight: '400px' }}>
              {loading && (
                 <Box sx={{ 
                     position: 'absolute', 
                     top: 0, 
                     left: 0, 
                     right: 0, 
                     bottom: 0, 
                     bgcolor: 'rgba(255,255,255,0.7)', 
                     backdropFilter: 'blur(2px)',
                     zIndex: 10,
                     display: 'flex', 
                     justifyContent: 'center', 
                     alignItems: 'center',
                     borderRadius: '12px'
                 }}>
                    <CircularProgress sx={{ color: '#2563eb' }} />
                 </Box>
              )}

              {report ? (
                <Paper 
                  ref={reportRef}
                  elevation={0} 
                  sx={{ 
                    p: 5, 
                    borderRadius: '12px', 
                    border: '1px solid #e2e8f0',
                    maxWidth: '900px',
                    mx: 'auto',
                    backgroundColor: '#ffffff',
                    minHeight: '600px' // Prevent collapse
                  }}
                >
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {report}
                    </ReactMarkdown>
                  </div>
                  {/* Footer for Export */}
                  <div className="mt-8 pt-4 border-t border-slate-100 text-center">
                      <Typography variant="caption" className="text-slate-400 font-mono text-[10px] tracking-widest uppercase">
                          {t('reports.footer_text')} • {new Date().getFullYear()}
                      </Typography>
                  </div>
                </Paper>
              ) : !loading && (
                <Box 
                  sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    height: '400px',
                    color: 'text.secondary',
                    border: '2px dashed #e2e8f0',
                    borderRadius: '12px',
                    bgcolor: '#f8fafc'
                  }}
                >
                  <ArticleIcon sx={{ fontSize: 48, mb: 2, color: '#94a3b8' }} />
                  <Typography variant="h6" sx={{ color: '#64748b', fontWeight: 600 }}>
                    {t('reports.select_prompt')}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                    {t('sentiment.select_sub')}
                  </Typography>
                </Box>
              )}
          </Box>
        </div>
      </div>
    </div>
  );
}