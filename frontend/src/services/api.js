const API_BASE_URL = 'http://127.0.0.1:8000/api';

class APIService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || errorData.error || `HTTP ${response.status}`);
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
    
    const token = localStorage.getItem('access_token');
    const headers = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    
    const response = await fetch(`${this.baseURL}/upload/`, {
      method: 'POST',
      headers: headers,
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || errorData.error || `HTTP ${response.status}`);
    }

    return await response.json();
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
    const refreshToken = localStorage.getItem('refresh_token');
    return this.request('/auth/logout/', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }
}

// Fix the ESLint warning by assigning to variable first
const apiService = new APIService();
export default apiService;