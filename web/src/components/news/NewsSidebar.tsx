import { useTranslation } from 'react-i18next';
import {
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Badge,
  Divider,
} from '@mui/material';
import ArticleIcon from '@mui/icons-material/Article';
import PublicIcon from '@mui/icons-material/Public';
import WbSunnyIcon from '@mui/icons-material/WbSunny';
import FlashOnIcon from '@mui/icons-material/FlashOn';
import CampaignIcon from '@mui/icons-material/Campaign';
import DescriptionIcon from '@mui/icons-material/Description';
import WhatshotIcon from '@mui/icons-material/Whatshot';
import BookmarkIcon from '@mui/icons-material/Bookmark';

import type { NewsCategory } from '../../pages/News';

interface NewsSidebarProps {
  category: NewsCategory;
  onCategoryChange: (category: NewsCategory) => void;
  showBookmarks: boolean;
  onShowBookmarks: () => void;
  bookmarksCount: number;
}

interface CategoryItem {
  id: NewsCategory;
  labelZh: string;
  labelEn: string;
  icon: React.ReactNode;
}

const categories: CategoryItem[] = [
  { id: 'all', labelZh: '全部资讯', labelEn: 'All News', icon: <ArticleIcon /> },
  { id: 'realtime', labelZh: '实时快讯', labelEn: 'Realtime', icon: <PublicIcon /> },
  { id: 'morning', labelZh: '财经早餐', labelEn: 'Morning Brief', icon: <WbSunnyIcon /> },
  { id: 'flash', labelZh: '自选股快讯', labelEn: 'Stock Flash', icon: <FlashOnIcon /> },
  { id: 'announcement', labelZh: '公告公示', labelEn: 'Announcements', icon: <CampaignIcon /> },
  { id: 'research', labelZh: '研报速递', labelEn: 'Research', icon: <DescriptionIcon /> },
  { id: 'hot', labelZh: '热门资讯', labelEn: 'Hot News', icon: <WhatshotIcon /> },
];

export default function NewsSidebar({
  category,
  onCategoryChange,
  showBookmarks,
  onShowBookmarks,
  bookmarksCount,
}: NewsSidebarProps) {
  const { t } = useTranslation();

  return (
    <Box sx={{ py: 1 }}>
      <List disablePadding>
        {categories.map((item) => (
          <ListItemButton
            key={item.id}
            selected={!showBookmarks && category === item.id}
            onClick={() => onCategoryChange(item.id)}
            sx={{
              mx: 1,
              borderRadius: 2,
              mb: 0.5,
              '&.Mui-selected': {
                backgroundColor: '#f1f5f9',
                '&:hover': {
                  backgroundColor: '#e2e8f0',
                },
              },
              '&:hover': {
                backgroundColor: '#f8fafc',
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 36, color: category === item.id ? '#6366f1' : '#64748b' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={t(`news.categories.${item.id}`)}
              primaryTypographyProps={{
                fontSize: '0.875rem',
                fontWeight: category === item.id ? 600 : 400,
                color: category === item.id ? '#0f172a' : '#64748b',
              }}
            />
          </ListItemButton>
        ))}
      </List>

      <Divider sx={{ my: 1.5, mx: 2 }} />

      <List disablePadding>
        <ListItemButton
          selected={showBookmarks}
          onClick={onShowBookmarks}
          sx={{
            mx: 1,
            borderRadius: 2,
            '&.Mui-selected': {
              backgroundColor: '#fef3c7',
              '&:hover': {
                backgroundColor: '#fde68a',
              },
            },
            '&:hover': {
              backgroundColor: '#fffbeb',
            },
          }}
        >
          <ListItemIcon sx={{ minWidth: 36, color: showBookmarks ? '#d97706' : '#64748b' }}>
            <Badge badgeContent={bookmarksCount} color="warning" max={99}>
              <BookmarkIcon />
            </Badge>
          </ListItemIcon>
          <ListItemText
            primary={t('news.sidebar.bookmarks')}
            primaryTypographyProps={{
              fontSize: '0.875rem',
              fontWeight: showBookmarks ? 600 : 400,
              color: showBookmarks ? '#92400e' : '#64748b',
            }}
          />
        </ListItemButton>
      </List>
    </Box>
  );
}
