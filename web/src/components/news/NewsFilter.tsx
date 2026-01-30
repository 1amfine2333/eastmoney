import { useTranslation } from 'react-i18next';
import {
  Box,
  TextField,
  InputAdornment,
  Chip,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ArticleIcon from '@mui/icons-material/Article';
import PublicIcon from '@mui/icons-material/Public';
import WbSunnyIcon from '@mui/icons-material/WbSunny';
import FlashOnIcon from '@mui/icons-material/FlashOn';
import CampaignIcon from '@mui/icons-material/Campaign';
import DescriptionIcon from '@mui/icons-material/Description';
import WhatshotIcon from '@mui/icons-material/Whatshot';

import type { NewsCategory } from '../../pages/News';

interface NewsFilterProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  category: NewsCategory;
  onCategoryChange: (category: NewsCategory) => void;
  timeRange: 'all' | '1d' | '3d' | '7d';
  onTimeRangeChange: (range: 'all' | '1d' | '3d' | '7d') => void;
  showCategoryChips?: boolean;
}

interface CategoryChip {
  id: NewsCategory;
  labelZh: string;
  labelEn: string;
  icon: React.ReactNode;
}

const categories: CategoryChip[] = [
  { id: 'all', labelZh: '全部', labelEn: 'All', icon: <ArticleIcon sx={{ fontSize: 16 }} /> },
  { id: 'realtime', labelZh: '实时', labelEn: 'Realtime', icon: <PublicIcon sx={{ fontSize: 16 }} /> },
  { id: 'morning', labelZh: '早餐', labelEn: 'Morning', icon: <WbSunnyIcon sx={{ fontSize: 16 }} /> },
  { id: 'flash', labelZh: '快讯', labelEn: 'Flash', icon: <FlashOnIcon sx={{ fontSize: 16 }} /> },
  { id: 'announcement', labelZh: '公告', labelEn: 'Announce', icon: <CampaignIcon sx={{ fontSize: 16 }} /> },
  { id: 'research', labelZh: '研报', labelEn: 'Research', icon: <DescriptionIcon sx={{ fontSize: 16 }} /> },
  { id: 'hot', labelZh: '热门', labelEn: 'Hot', icon: <WhatshotIcon sx={{ fontSize: 16 }} /> },
];

export default function NewsFilter({
  searchQuery,
  onSearchChange,
  category,
  onCategoryChange,
  timeRange,
  onTimeRangeChange,
  showCategoryChips = false,
}: NewsFilterProps) {
  const { t } = useTranslation();

  return (
    <Box sx={{ p: 2, borderBottom: '1px solid #f1f5f9' }}>
      {/* Search */}
      <TextField
        fullWidth
        size="small"
        placeholder={t('news.filter.search')}
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon sx={{ color: '#94a3b8', fontSize: 20 }} />
            </InputAdornment>
          ),
        }}
        sx={{
          '& .MuiOutlinedInput-root': {
            backgroundColor: '#f8fafc',
            borderRadius: 2,
            '& fieldset': {
              borderColor: '#e2e8f0',
            },
            '&:hover fieldset': {
              borderColor: '#cbd5e1',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6366f1',
            },
          },
        }}
      />

      {/* Time range */}
      <Box sx={{ display: 'flex', gap: 1, mt: 1.5, flexWrap: 'wrap' }}>
        {([
          { id: '1d', zh: '24h', en: '24h' },
          { id: '3d', zh: '3天', en: '3d' },
          { id: '7d', zh: '7天', en: '7d' },
          { id: 'all', zh: '全部', en: 'All' },
        ] as const).map((r) => (
          <Chip
            key={r.id}
            label={t(`news.timerange.${r.id}`)}
            onClick={() => onTimeRangeChange(r.id)}
            variant={timeRange === r.id ? 'filled' : 'outlined'}
            sx={{
              backgroundColor: timeRange === r.id ? '#0ea5e9' : 'transparent',
              color: timeRange === r.id ? '#fff' : '#64748b',
              borderColor: '#e2e8f0',
              '&:hover': {
                backgroundColor: timeRange === r.id ? '#0284c7' : '#f1f5f9',
              },
            }}
          />
        ))}
      </Box>

      {/* Category chips (for mobile) */}
      {showCategoryChips && (
        <Box
          sx={{
            display: 'flex',
            gap: 1,
            mt: 1.5,
            overflowX: 'auto',
            pb: 0.5,
            '&::-webkit-scrollbar': {
              display: 'none',
            },
          }}
        >
          {categories.map((item) => (
            <Chip
              key={item.id}
              label={t(`news.filter.${item.id}`)}
              icon={item.icon as React.ReactElement}
              onClick={() => onCategoryChange(item.id)}
              variant={category === item.id ? 'filled' : 'outlined'}
              sx={{
                flexShrink: 0,
                backgroundColor: category === item.id ? '#6366f1' : 'transparent',
                color: category === item.id ? '#fff' : '#64748b',
                borderColor: '#e2e8f0',
                '&:hover': {
                  backgroundColor: category === item.id ? '#4f46e5' : '#f1f5f9',
                },
                '& .MuiChip-icon': {
                  color: category === item.id ? '#fff' : '#64748b',
                },
              }}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}
