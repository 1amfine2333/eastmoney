import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Box, Typography, useTheme } from '@mui/material';

interface AnimatedNumberProps {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  duration?: number;
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'success' | 'error' | 'inherit';
  showSign?: boolean;
}

const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  prefix = '',
  suffix = '',
  decimals = 2,
  duration = 0.5,
  size = 'medium',
  color = 'inherit',
  showSign = false,
}) => {
  const theme = useTheme();
  const [displayValue, setDisplayValue] = React.useState(0);
  const previousValue = React.useRef(0);

  React.useEffect(() => {
    const startValue = previousValue.current;
    const endValue = value;
    const startTime = Date.now();
    const durationMs = duration * 1000;
    let animationId: number;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / durationMs, 1);

      // Easing function (ease-out cubic)
      const eased = 1 - Math.pow(1 - progress, 3);

      const currentValue = startValue + (endValue - startValue) * eased;
      setDisplayValue(currentValue);

      if (progress < 1) {
        animationId = requestAnimationFrame(animate);
      } else {
        previousValue.current = endValue;
      }
    };

    animationId = requestAnimationFrame(animate);

    // Cleanup: cancel animation frame on unmount or value change
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [value, duration]);

  const fontSize = {
    small: '1rem',
    medium: '1.5rem',
    large: '2.5rem',
  }[size];

  const fontWeight = {
    small: 500,
    medium: 600,
    large: 700,
  }[size];

  const getColor = () => {
    if (color === 'inherit') return 'inherit';
    if (color === 'success') return theme.palette.success.main;
    if (color === 'error') return theme.palette.error.main;
    return theme.palette.primary.main;
  };

  const formattedValue = displayValue.toFixed(decimals);
  const sign = showSign && value > 0 ? '+' : '';

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={value}
        initial={{ opacity: 0.5, y: -5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <Typography
          component="span"
          sx={{
            fontSize,
            fontWeight,
            color: getColor(),
            fontFamily: '"Roboto Mono", monospace',
            letterSpacing: '-0.5px',
          }}
        >
          {prefix}{sign}{formattedValue}{suffix}
        </Typography>
      </motion.div>
    </AnimatePresence>
  );
};

export default AnimatedNumber;
