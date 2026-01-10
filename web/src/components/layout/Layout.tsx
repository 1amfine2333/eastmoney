import { useState, useEffect } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { 
  Drawer, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  Typography, 
  Box,
  Divider,
  Avatar,
  Tooltip,
  IconButton
} from '@mui/material';
import PieChartIcon from '@mui/icons-material/PieChart';
import ArticleIcon from '@mui/icons-material/Article';
import SettingsIcon from '@mui/icons-material/Settings';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
import WifiTetheringIcon from '@mui/icons-material/WifiTethering';
import PublicIcon from '@mui/icons-material/Public';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import SpeedIcon from '@mui/icons-material/Speed';

import { fetchMarketIndices,  } from '../../api';
import type {IndexData} from '../../api';

const drawerWidth = 260;

const MENU_ITEMS = [
  { text: 'Dashboard', icon: <SpeedIcon />, path: '/dashboard', subtitle: 'Market cockpit' },
  { text: 'Universe', icon: <PieChartIcon />, path: '/funds', subtitle: 'Manage portfolio nodes' },
  { text: 'Sentiment', icon: <AutoGraphIcon />, path: '/sentiment', subtitle: 'Market vibe analysis' },
  { text: 'Intelligence', icon: <ArticleIcon />, path: '/reports', subtitle: 'Deep dive reports' },
  { text: 'Commodities', icon: <MonetizationOnIcon />, path: '/commodities', subtitle: 'Global assets' },
  { text: 'System', icon: <SettingsIcon />, path: '/settings', subtitle: 'Core configuration' },
];

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const [indices, setIndices] = useState<IndexData[]>([]);

  useEffect(() => {
    const loadIndices = async () => {
      try {
        const data = await fetchMarketIndices();
        setIndices(data);
      } catch (err) {
        console.error("Failed to load market indices", err);
      }
    };
    
    loadIndices();
    const timer = setInterval(loadIndices, 60000); // Refresh every 60s
    return () => clearInterval(timer);
  }, []);

  const currentItem = MENU_ITEMS.find(item => location.pathname.startsWith(item.path)) || MENU_ITEMS[0];

  return (
    <div className="flex min-h-screen bg-white text-slate-900 font-sans">
      {/* Navigation Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { 
            width: drawerWidth, 
            boxSizing: 'border-box',
            borderRight: '1px solid #f1f5f9',
            backgroundColor: '#ffffff',
          },
        }}
      >
        <div className="h-20 flex items-center px-6 border-b border-slate-50">
          <img src="/vite.svg" alt="Logo" className="w-8 h-8 mr-3 shadow-sm rounded-xl" />
          <div className="flex flex-col">
            <Typography variant="h6" className="tracking-wide font-bold text-slate-900 leading-none" sx={{ fontFamily: 'JetBrains Mono' }}>
              V<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-600">ALPHA</span>
            </Typography>
            <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700, fontSize: '0.6rem', mt: 0.5, letterSpacing: '0.1em' }}>
              INTELLIGENCE CORE
            </Typography>
          </div>
        </div>
        
        <div className="flex-1 py-6 px-3 overflow-y-auto">
          <List className="space-y-1">
            {MENU_ITEMS.map((item) => {
               const isActive = location.pathname.startsWith(item.path);
               return (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton 
                    selected={isActive}
                    onClick={() => navigate(item.path)}
                    sx={{
                      borderRadius: '12px',
                      py: 1.5,
                      '&.Mui-selected': {
                        backgroundColor: '#f8fafc',
                        '&:hover': { backgroundColor: '#f1f5f9' },
                        '&::before': {
                            content: '""',
                            position: 'absolute',
                            left: -12,
                            width: 4,
                            height: 24,
                            borderRadius: '0 4px 4px 0',
                            bgcolor: '#6366f1'
                        }
                      },
                      '&:hover': { backgroundColor: '#f8fafc' }
                    }}
                  >
                    <ListItemIcon sx={{ 
                      minWidth: 40, 
                      color: isActive ? '#6366f1' : '#94a3b8' 
                    }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText 
                      primary={item.text} 
                      primaryTypographyProps={{ 
                        fontWeight: isActive ? 800 : 600,
                        fontSize: '0.85rem',
                        color: isActive ? '#0f172a' : '#64748b'
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </div>

        <Box sx={{ p: 3, borderTop: '1px solid #f1f5f9' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1.5, bgcolor: '#f8fafc', borderRadius: '12px' }}>
                <Avatar sx={{ width: 32, height: 32, bgcolor: '#6366f1', fontSize: '0.8rem', fontWeight: 800 }}>A</Avatar>
                <Box>
                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 800, color: '#1e293b' }}>Administrator</Typography>
                    <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', fontWeight: 600 }}>System Master</Typography>
                </Box>
            </Box>
        </Box>
      </Drawer>

      {/* Main Content Area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', bgcolor: '#fcfcfc' }}>
        
        {/* Modern Header */}
        <Box component="header" sx={{ 
            height: 70, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            px: 4, 
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(12px)',
            borderBottom: '1px solid #f1f5f9',
            zIndex: 10
        }}>
            {/* Left: Page Identity */}
            <Box sx={{ width: 240, flexShrink: 0 }}>
                <Typography noWrap sx={{ color: '#0f172a', fontWeight: 900, fontSize: '1.1rem', letterSpacing: '-0.02em' }}>
                    {currentItem.text}
                </Typography>
                <Typography noWrap sx={{ color: '#94a3b8', fontSize: '0.75rem', fontWeight: 600 }}>
                    {currentItem.subtitle}
                </Typography>
            </Box>

            {/* Right: Indices & System Tools */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                {/* Real-time Indices with Fixed Width to prevent jitter */}
                <Box sx={{ display: 'flex', gap: 1 }}>
                    {indices.map((idx, i) => {
                        const isUp = idx.change_pct >= 0;
                        const color = isUp ? '#ef4444' : '#22c55e';
                        return (
                            <Box key={idx.code} sx={{ 
                                display: 'flex', 
                                alignItems: 'center',
                                gap: 1.5,
                                px: 2,
                                borderRight: i < indices.length - 1 ? '1px solid #f1f5f9' : 'none',
                                width: '140px' // Fixed width to prevent layout shift
                            }}>
                                <Box>
                                    <Typography sx={{ fontSize: '0.65rem', fontWeight: 800, color: '#94a3b8', whiteSpace: 'nowrap' }}>{idx.name}</Typography>
                                    <Typography sx={{ fontSize: '0.9rem', fontWeight: 900, fontFamily: 'JetBrains Mono', color: '#1e293b', lineHeight: 1.2 }}>
                                        {idx.price.toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box sx={{ 
                                    display: 'flex', 
                                    flexDirection: 'column', 
                                    alignItems: 'flex-end',
                                    color: color,
                                    minWidth: '50px'
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                        {isUp ? <ArrowDropUpIcon sx={{ fontSize: 18 }} /> : <ArrowDropDownIcon sx={{ fontSize: 18 }} />}
                                        <Typography sx={{ fontSize: '0.75rem', fontWeight: 900, fontFamily: 'JetBrains Mono' }}>
                                            {Math.abs(idx.change_pct).toFixed(2)}%
                                        </Typography>
                                    </Box>
                                    <Typography sx={{ fontSize: '0.6rem', fontWeight: 700, fontFamily: 'JetBrains Mono', mt: -0.5 }}>
                                        {isUp ? '+' : ''}{idx.change_val.toFixed(2)}
                                    </Typography>
                                </Box>
                            </Box>
                        )
                    })}
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Divider orientation="vertical" flexItem sx={{ height: 24, mx: 1, borderColor: '#f1f5f9' }} />
                    <Tooltip title="Market Network Sync">
                        <IconButton size="small" sx={{ color: '#94a3b8', '&:hover': { color: '#6366f1', bgcolor: '#f8fafc' } }}>
                            <PublicIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Core Terminal v2.1">
                        <IconButton size="small" sx={{ color: '#94a3b8', '&:hover': { color: '#6366f1', bgcolor: '#f8fafc' } }}>
                            <WifiTetheringIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>
        </Box>

        {/* Content Section */}
        <Box component="main" sx={{ flexGrow: 1, overflow: 'auto', p: 4 }}>
          <Outlet />
        </Box>

        {/* Minimal Footer */}
        <Box component="footer" sx={{ 
            py: 2, 
            px: 4, 
            bgcolor: '#ffffff', 
            borderTop: '1px solid #f1f5f9',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 600 }}>
                    © 2026 VibeAlpha Terminal.
                </Typography>
                <Typography sx={{ color: '#cbd5e1', fontSize: '0.7rem' }}>|</Typography>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    Data by <span className="text-indigo-400 font-bold">AkShare & Xueqiu</span> <InfoOutlinedIcon sx={{ fontSize: 12 }} />
                </Typography>
            </Box>
        </Box>
      </Box>

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
          100% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}