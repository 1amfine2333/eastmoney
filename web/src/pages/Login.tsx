import { useState } from 'react';
import { 
    Box, 
    Button, 
    TextField, 
    Typography, 
    Paper, 
    Tabs, 
    Tab, 
    InputAdornment, 
    IconButton,
    CircularProgress,
    Snackbar,
    Alert
} from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import LockIcon from '@mui/icons-material/Lock';
import PersonIcon from '@mui/icons-material/Person';
import EmailIcon from '@mui/icons-material/Email';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { login, register } from '../api';

export default function LoginPage() {
    const [tab, setTab] = useState(0);
    const [loading, setLoading] = useState(false);
    
    // Form State
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [email, setEmail] = useState('');
    const [showPassword, setShowPassword] = useState(false);

    // Notification State
    const [notify, setNotify] = useState<{ open: boolean, message: string, severity: 'success' | 'error' }>({
        open: false,
        message: '',
        severity: 'success'
    });

    const handleAuth = async () => {
        if (!username || !password) {
            setNotify({ open: true, message: 'Username and Password are required', severity: 'error' });
            return;
        }

        setLoading(true);
        try {
            let res;
            if (tab === 0) {
                // Login
                res = await login(username, password);
            } else {
                // Register
                res = await register(username, password, email);
            }
            
            // Success
            localStorage.setItem('token', res.access_token);
            window.location.href = '/dashboard'; // Hard reload to clear state/init app
            
        } catch (error: any) {
            console.error(error);
            let msg = 'Authentication failed';
            if (error.response?.data?.detail) {
                msg = error.response.data.detail;
            }
            setNotify({ open: true, message: msg, severity: 'error' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={{ 
            height: '100vh', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            bgcolor: '#f8fafc',
            backgroundImage: 'radial-gradient(#e2e8f0 1px, transparent 1px)',
            backgroundSize: '30px 30px'
        }}>
            <Paper elevation={0} sx={{ 
                width: 400, 
                p: 4, 
                borderRadius: '24px', 
                border: '1px solid #f1f5f9',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)'
            }}>
                {/* Brand Header */}
                <Box sx={{ textAlign: 'center', mb: 4 }}>
                    <Box sx={{ 
                        display: 'inline-flex', 
                        p: 1.5, 
                        borderRadius: '16px', 
                        bgcolor: '#e0e7ff', 
                        color: '#4338ca',
                        mb: 2
                    }}>
                        <TrendingUpIcon fontSize="large" />
                    </Box>
                    <Typography variant="h5" sx={{ fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em' }}>
                        Lumina Alpha
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#64748b', mt: 0.5 }}>
                        Market Intelligence Terminal
                    </Typography>
                </Box>

                {/* Tabs */}
                <Tabs 
                    value={tab} 
                    onChange={(_, v) => setTab(v)} 
                    variant="fullWidth" 
                    sx={{ 
                        mb: 3, 
                        borderBottom: '1px solid #f1f5f9',
                        '& .MuiTab-root': { fontWeight: 700, textTransform: 'none', fontSize: '0.9rem' },
                        '& .Mui-selected': { color: '#4338ca' },
                        '& .MuiTabs-indicator': { bgcolor: '#4338ca', height: 3, borderRadius: '3px 3px 0 0' }
                    }}
                >
                    <Tab label="Sign In" />
                    <Tab label="Create Account" />
                </Tabs>

                {/* Form */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                    <TextField
                        fullWidth
                        label="Username"
                        variant="outlined"
                        size="medium"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><PersonIcon sx={{ color: '#94a3b8' }} /></InputAdornment>,
                        }}
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
                    />
                    
                    {tab === 1 && (
                        <TextField
                            fullWidth
                            label="Email (Optional)"
                            variant="outlined"
                            size="medium"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            InputProps={{
                                startAdornment: <InputAdornment position="start"><EmailIcon sx={{ color: '#94a3b8' }} /></InputAdornment>,
                            }}
                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
                        />
                    )}

                    <TextField
                        fullWidth
                        label="Password"
                        type={showPassword ? 'text' : 'password'}
                        variant="outlined"
                        size="medium"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleAuth()}
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><LockIcon sx={{ color: '#94a3b8' }} /></InputAdornment>,
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton onClick={() => setShowPassword(!showPassword)} edge="end">
                                        {showPassword ? <VisibilityOff /> : <Visibility />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
                    />

                    <Button
                        fullWidth
                        variant="contained"
                        size="large"
                        onClick={handleAuth}
                        disabled={loading}
                        sx={{
                            mt: 1,
                            bgcolor: '#4338ca',
                            borderRadius: '12px',
                            py: 1.5,
                            textTransform: 'none',
                            fontSize: '1rem',
                            fontWeight: 700,
                            boxShadow: '0 4px 6px -1px rgba(67, 56, 202, 0.4)',
                            '&:hover': { bgcolor: '#3730a3', boxShadow: '0 10px 15px -3px rgba(67, 56, 202, 0.4)' }
                        }}
                    >
                        {loading ? <CircularProgress size={24} color="inherit" /> : (tab === 0 ? 'Sign In' : 'Register')}
                    </Button>
                </Box>

                {/* Footer */}
                <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 4, color: '#94a3b8' }}>
                    © {new Date().getFullYear()} Lumina Brain. All rights reserved.
                </Typography>
            </Paper>

            <Snackbar 
                open={notify.open} 
                autoHideDuration={4000} 
                onClose={() => setNotify(prev => ({ ...prev, open: false }))}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                <Alert severity={notify.severity} sx={{ width: '100%', borderRadius: '10px', fontWeight: 600 }}>
                    {notify.message}
                </Alert>
            </Snackbar>
        </Box>
    );
}
