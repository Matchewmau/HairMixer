class AuthService {
  constructor() {
    // Unify default base URL with APIService; allow override via env
    this.baseURL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';
    // Feature flag: enable HttpOnly cookie-based auth
    this.useCookieAuth = (
      String(process.env.REACT_APP_USE_COOKIE_AUTH || 'false').toLowerCase() === 'true'
    );
  }

  async login(credentials) {
    try {
      const response = await fetch(`${this.baseURL}/auth/login/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: this.useCookieAuth ? 'include' : 'same-origin',
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Login failed');
      }

      // Cookie mode: do not persist tokens client-side
      if (!this.useCookieAuth) {
        // Store tokens in localStorage (both camelCase and snake_case)
        if (data.access_token) {
          localStorage.setItem('accessToken', data.access_token);
          localStorage.setItem('access_token', data.access_token);
        }
        if (data.refresh_token) {
          localStorage.setItem('refreshToken', data.refresh_token);
          localStorage.setItem('refresh_token', data.refresh_token);
        }
      }
      if (data.user) {
        localStorage.setItem('user', JSON.stringify(data.user));
      }

      return { success: true, user: data.user };
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  async signup(userData) {
    try {
      const response = await fetch(`${this.baseURL}/auth/signup/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: this.useCookieAuth ? 'include' : 'same-origin',
        body: JSON.stringify(userData),
      });

      const data = await response.json();

      if (!response.ok) {
        const nested = (data && data.error) || {};
        const msg = nested.message || data.message || (typeof data.error === 'string' ? data.error : '') || 'Registration failed';
        throw new Error(msg);
      }

      if (!this.useCookieAuth) {
        if (data.access_token) {
          localStorage.setItem('accessToken', data.access_token);
          localStorage.setItem('access_token', data.access_token);
        }
        if (data.refresh_token) {
          localStorage.setItem('refreshToken', data.refresh_token);
          localStorage.setItem('refresh_token', data.refresh_token);
        }
      }
      if (data.user) {
        localStorage.setItem('user', JSON.stringify(data.user));
      }

      return { success: true, user: data.user };
    } catch (error) {
      console.error('Signup error:', error);
      throw error;
    }
  }

  async logout() {
    try {
      const token = this.getAccessToken();
      const refreshToken = this.getRefreshToken();

      await fetch(`${this.baseURL}/auth/logout/`, {
        method: 'POST',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          'Content-Type': 'application/json',
        },
        credentials: this.useCookieAuth ? 'include' : 'same-origin',
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless of API call success
      localStorage.removeItem('accessToken');
      localStorage.removeItem('access_token');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
    }
  }

  async getCurrentUser() {
    try {
      const token = this.getAccessToken();
      const storedUser = localStorage.getItem('user');

      // In cookie mode, token may be absent; rely on server session
      if (!this.useCookieAuth && !token) return null;

      if (storedUser) {
        return JSON.parse(storedUser);
      }

      // If no stored user, fetch from API
      const response = await fetch(`${this.baseURL}/auth/profile/`, {
        headers: this.useCookieAuth ? {} : { 'Authorization': `Bearer ${token}` },
        credentials: this.useCookieAuth ? 'include' : 'same-origin',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch user');
      }

      const payload = await response.json();
      const user = payload && payload.user ? payload.user : null;
      if (user) {
        localStorage.setItem('user', JSON.stringify(user));
      }
      return user;
    } catch (error) {
      console.error('Get current user error:', error);
      // Do not eagerly clear tokens here; caller may decide. If unauthorized, logout below.
      return null;
    }
  }

  getAccessToken() {
    return localStorage.getItem('accessToken');
  }

  getRefreshToken() {
    return localStorage.getItem('refreshToken');
  }

  isAuthenticated() {
    if (this.useCookieAuth) {
      // Heuristic: if we have a cached user, treat as authenticated
      const u = localStorage.getItem('user');
      return !!u;
    }
    return !!this.getAccessToken();
  }

  async refreshAccessToken() {
    try {
      if (this.useCookieAuth) {
        // Cookie mode: rely on server session; no client-side refresh
        throw new Error('Cookie-based auth does not support client refresh');
      }
      const refreshToken = this.getRefreshToken();
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await fetch(`${this.baseURL}/auth/refresh/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // SimpleJWT expects { refresh: <token> }
        body: JSON.stringify({ refresh: refreshToken }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      // SimpleJWT returns { access: <newAccessToken> }
      const newAccess = data.access || data.access_token;
      if (!newAccess) {
        throw new Error('Token refresh response missing access token');
      }
      localStorage.setItem('accessToken', newAccess);
      localStorage.setItem('access_token', newAccess);
      return newAccess;
    } catch (error) {
      console.error('Token refresh error:', error);
      this.logout();
      throw error;
    }
  }

  // HTTP interceptor for API calls with automatic token refresh
  async apiCall(url, options = {}) {
    let token = this.getAccessToken();

    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    if (!this.useCookieAuth && token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        credentials: this.useCookieAuth ? 'include' : (options.credentials || 'same-origin'),
      });

      // If token expired, try to refresh
      if (!this.useCookieAuth && response.status === 401 && this.getRefreshToken()) {
        token = await this.refreshAccessToken();
        headers.Authorization = `Bearer ${token}`;
        
        return fetch(url, {
          ...options,
          headers,
          credentials: options.credentials || 'same-origin',
        });
      }

      return response;
    } catch (error) {
      console.error('API call error:', error);
      throw error;
    }
  }
}

const authService = new AuthService();
export default authService;