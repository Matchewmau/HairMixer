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
      signal: options.signal, // Add signal for AbortController
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
        const nested = (errorData && errorData.error) || {};
        const msg = nested.message || errorData.message || errorData.detail || (typeof errorData.error === 'string' ? errorData.error : '') || `HTTP ${response.status}`;
        const error = new Error(msg || 'Request failed');
        error.status = response.status;
        error.data = errorData;
        throw error;
      }
      
      // Handle 204 No Content responses (e.g., DELETE operations)
      if (response.status === 204) {
        return null;
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

  // ML-based hairstyle recommendations (top 10)
  async getMLRecommendations(preferenceId) {
    return this.request('/recommend/ml/', {
      method: 'POST',
      body: JSON.stringify({
        preference_id: preferenceId,
      }),
    });
  }

  // Overlay generation
  async generateOverlay(imageId, hairstyleId, overlayType = 'advanced', signal = null, useHairColor = false) {
    return this.request('/overlay/', {
      method: 'POST',
      body: JSON.stringify({
        image_id: imageId,
        hairstyle_id: hairstyleId,
        overlay_type: overlayType,
        use_hair_color: useHairColor,
      }),
      signal: signal, // Add signal for AbortController
    });
  }

  // Get detailed hairstyle information with AI-generated content
  async getHairstyleDetailsWithAI(hairstyleId, preferenceId = null, imageId = null) {
    const params = new URLSearchParams();
    if (preferenceId) params.append('preference_id', preferenceId);
    if (imageId) params.append('image_id', imageId);
    const query = params.toString();
    return this.request(`/hairstyles/${hairstyleId}/details/${query ? '?' + query : ''}`);
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

  // Preference Profiles
  async getPreferenceProfiles() {
    return this.request('/preference-profiles/');
  }

  async createPreferenceProfile(profileData) {
    return this.request('/preference-profiles/', {
      method: 'POST',
      body: JSON.stringify(profileData),
    });
  }

  async getPreferenceProfile(profileId) {
    return this.request(`/preference-profiles/${profileId}/`);
  }

  async updatePreferenceProfile(profileId, profileData) {
    return this.request(`/preference-profiles/${profileId}/`, {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }

  async deletePreferenceProfile(profileId) {
    return this.request(`/preference-profiles/${profileId}/`, {
      method: 'DELETE',
    });
  }

  async setDefaultProfile(profileId) {
    return this.request(`/preference-profiles/${profileId}/set-default/`, {
      method: 'POST',
    });
  }

  // Saved Hairstyles
  async getSavedHairstyles() {
    return this.request('/saved-hairstyles/');
  }

  async saveHairstyle(hairstyleData) {
    return this.request('/saved-hairstyles/', {
      method: 'POST',
      body: JSON.stringify(hairstyleData),
    });
  }

  async getSavedHairstyle(savedId) {
    return this.request(`/saved-hairstyles/${savedId}/`);
  }

  async updateSavedHairstyle(savedId, hairstyleData) {
    return this.request(`/saved-hairstyles/${savedId}/`, {
      method: 'PUT',
      body: JSON.stringify(hairstyleData),
    });
  }

  async deleteSavedHairstyle(savedId) {
    return this.request(`/saved-hairstyles/${savedId}/`, {
      method: 'DELETE',
    });
  }

  // Hairstyle Likes/Dislikes
  async likeHairstyle(hairstyleId, reaction) {
    return this.request('/hairstyle-likes/', {
      method: 'POST',
      body: JSON.stringify({
        hairstyle_id: hairstyleId,
        reaction: reaction, // 'like' or 'dislike'
      }),
    });
  }

  async removeHairstyleLike(hairstyleId) {
    return this.request('/hairstyle-likes/', {
      method: 'DELETE',
      body: JSON.stringify({
        hairstyle_id: hairstyleId,
      }),
    });
  }

  async getHairstyleLikeStats(hairstyleId) {
    return this.request(`/hairstyle-likes/stats/${hairstyleId}/`);
  }

  async getHairstyleLikeBulkStats(hairstyleIds) {
    return this.request('/hairstyle-likes/bulk-stats/', {
      method: 'POST',
      body: JSON.stringify({
        hairstyle_ids: hairstyleIds,
      }),
    });
  }

  async getUserLikedHairstyles(reaction = null) {
    const query = reaction ? `?reaction=${reaction}` : '';
    return this.request(`/hairstyle-likes/user/${query}`);
  }

  // User Profile
  async getUserProfile() {
    return this.request('/auth/profile/');
  }

  async updateUserProfile(profileData) {
    return this.request('/auth/profile/', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }
}

// Fix the ESLint warning by assigning to variable first
const apiService = new APIService();
export default apiService;