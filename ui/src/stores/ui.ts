/**
 * UI state management store using Zustand
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Notification interface
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number; // ms, 0 for persistent
  timestamp: number;
  jobId?: string;
}

// Theme type
export type Theme = 'light' | 'dark' | 'system';

interface UIState {
  // Theme
  theme: Theme;

  // Sidebar
  sidebarCollapsed: boolean;

  // Notifications
  notifications: Notification[];

  // Connection status indicator
  showConnectionStatus: boolean;

  // Auto-scroll for logs
  autoScrollLogs: boolean;
}

interface UIActions {
  // Theme
  setTheme: (theme: Theme) => void;

  // Sidebar
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Notifications
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  dismissNotification: (id: string) => void;
  clearAllNotifications: () => void;

  // Shortcuts
  addJobNotification: (jobId: string, status: 'completed' | 'failed' | 'stopped') => void;

  // Settings
  setShowConnectionStatus: (show: boolean) => void;
  setAutoScrollLogs: (auto: boolean) => void;
}

export const useUIStore = create<UIState & UIActions>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        theme: 'system',
        sidebarCollapsed: false,
        notifications: [],
        showConnectionStatus: true,
        autoScrollLogs: true,

        // Theme actions
        setTheme: (theme) => set({ theme }),

        // Sidebar actions
        toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

        // Notification actions
        addNotification: (notification) => {
          const id = `notif-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
          const newNotification: Notification = {
            ...notification,
            id,
            timestamp: Date.now(),
            duration: notification.duration ?? 5000,
          };

          set((state) => ({
            notifications: [...state.notifications, newNotification],
          }));

          // Auto-dismiss if duration is set
          if (newNotification.duration && newNotification.duration > 0) {
            setTimeout(() => {
              get().dismissNotification(id);
            }, newNotification.duration);
          }

          return id;
        },

        dismissNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          })),

        clearAllNotifications: () => set({ notifications: [] }),

        // Job notification shortcut
        addJobNotification: (jobId, status) => {
          const notifications: Record<string, { type: Notification['type']; title: string; message: string }> = {
            completed: {
              type: 'success',
              title: 'Training Complete',
              message: `Job ${jobId.substring(0, 12)} completed successfully`,
            },
            failed: {
              type: 'error',
              title: 'Training Failed',
              message: `Job ${jobId.substring(0, 12)} encountered an error`,
            },
            stopped: {
              type: 'warning',
              title: 'Training Stopped',
              message: `Job ${jobId.substring(0, 12)} was stopped`,
            },
          };

          const notif = notifications[status];
          if (notif) {
            get().addNotification({
              ...notif,
              jobId,
              duration: status === 'completed' ? 5000 : 10000,
            });
          }
        },

        // Settings
        setShowConnectionStatus: (show) => set({ showConnectionStatus: show }),
        setAutoScrollLogs: (auto) => set({ autoScrollLogs: auto }),
      }),
      {
        name: 'leap-ui-storage',
        partialize: (state) => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
          showConnectionStatus: state.showConnectionStatus,
          autoScrollLogs: state.autoScrollLogs,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);
