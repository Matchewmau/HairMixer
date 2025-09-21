import AuthService from './AuthService';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

class APIService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      body: options.body,
    };

    try {
      // Use AuthService for authenticated calls with auto-refresh; fall back for public endpoints
      const hasAuth = !!AuthService.getAccessToken();
      const response = hasAuth
        ? await AuthService.apiCall(url, config)
        : await fetch(url, config);
      
      if (!response.ok) {
        let errorData = {};
        try {
          errorData = await response.json();
        } catch (_) {
          // ignore
        }
        const msg = errorData.message || errorData.detail || errorData.error || `HTTP ${response.status}`;
        const error = new Error(msg);
        error.status = response.status;
        error.data = errorData;
        throw error;
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.request('/health/');
  }

  // Face shapes and filters
  async getFaceShapes() {
    return this.request('/filter/face-shapes/');
  }

  async getOccasions() {
    return this.request('/filter/occasions/');
  }

  // Hairstyles
  async getHairstyles(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/hairstyles/${query ? '?' + query : ''}`);
  }

  async getFeaturedHairstyles() {
    return this.request('/hairstyles/featured/');
  }

  async searchHairstyles(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/search/${query ? '?' + query : ''}`);
  }

  // Image upload
  async uploadImage(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const headers = {}; // Do not set content-type for FormData
    const token = AuthService.getAccessToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(`${this.baseURL}/upload/`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!resp.ok) {
      let errorData = {};
      try { errorData = await resp.json(); } catch (_) {}
      const msg = errorData.message || errorData.error || `HTTP ${resp.status}`;
      const error = new Error(msg);
      error.status = resp.status;
      error.data = errorData;
      throw error;
    }

    return await resp.json();
  }

  // Preferences
  async savePreferences(preferences) {
    return this.request('/preferences/', {
      method: 'POST',
      body: JSON.stringify(preferences),
    });
  }

  // Recommendations
  async getRecommendations(imageId, preferenceId) {
    return this.request('/recommend/', {
      method: 'POST',
      body: JSON.stringify({
        image_id: imageId,
        preference_id: preferenceId,
      }),
    });
  }

  // Overlay generation
  async generateOverlay(imageId, hairstyleId, overlayType = 'basic') {
    return this.request('/overlay/', {
      method: 'POST',
      body: JSON.stringify({
        image_id: imageId,
        hairstyle_id: hairstyleId,
        overlay_type: overlayType,
      }),
    });
  }

  // Feedback
  async submitFeedback(feedbackData) {
    return this.request('/feedback/', {
      method: 'POST',
      body: JSON.stringify(feedbackData),
    });
  }

  // Authentication
  async signup(userData) {
    return this.request('/auth/signup/', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async login(credentials) {
    return this.request('/auth/login/', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  async logout() {
    const refreshToken = AuthService.getRefreshToken() || localStorage.getItem('refresh_token');
    return this.request('/auth/logout/', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }
}

// Fix the ESLint warning by assigning to variable first
const apiService = new APIService();
export default apiService;