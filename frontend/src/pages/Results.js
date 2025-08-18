import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { preferences, imageFile, previewUrl, uploadResponse, recommendations } = location.state || {};

  if (!recommendations) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            No recommendations found
          </h2>
          <button
            onClick={() => navigate('/upload')}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            Start Over
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Your Hairstyle Recommendations
          </h1>
          {previewUrl && (
            <div className="flex justify-center mb-4">
              <img
                src={previewUrl}
                alt="Your photo"
                className="h-32 w-32 object-cover rounded-full border-4 border-white shadow-lg"
              />
            </div>
          )}
          <p className="text-lg text-gray-600">
            Based on your preferences and facial analysis
          </p>
        </div>

        {/* Face Analysis Summary */}
        {uploadResponse && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Face Analysis</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="bg-blue-100 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900">Face Shape</h3>
                  <p className="text-blue-600 font-semibold">
                    {uploadResponse.face_shape?.shape || 'Analyzing...'}
                  </p>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-green-100 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900">Confidence</h3>
                  <p className="text-green-600 font-semibold">
                    {uploadResponse.face_shape?.confidence ? 
                      `${Math.round(uploadResponse.face_shape.confidence * 100)}%` : 
                      'N/A'}
                  </p>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-purple-100 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900">Quality Score</h3>
                  <p className="text-purple-600 font-semibold">
                    {uploadResponse.quality_score ? 
                      `${Math.round(uploadResponse.quality_score * 10)}/10` : 
                      'N/A'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Recommended Hairstyles</h2>
          
          {recommendations.recommended_styles && recommendations.recommended_styles.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.recommended_styles.map((style, index) => (
                <div key={index} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                  {style.image_url && (
                    <img
                      src={style.image_url}
                      alt={style.name}
                      className="w-full h-48 object-cover rounded-lg mb-4"
                    />
                  )}
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">
                    {style.name}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    {style.description}
                  </p>
                  <div className="flex justify-between items-center mb-4">
                    <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded">
                      {style.match_score ? `${Math.round(style.match_score * 100)}% Match` : 'Recommended'}
                    </span>
                    <span className="text-sm text-gray-500">
                      {style.difficulty || 'Medium'} Difficulty
                    </span>
                  </div>
                  <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors">
                    Try This Style
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-600 mb-4">
                No specific recommendations available yet, but here are some popular styles for your face shape:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Placeholder recommendations */}
                {[
                  { name: 'Classic Bob', description: 'A timeless bob cut that suits most face shapes' },
                  { name: 'Beach Waves', description: 'Relaxed, natural-looking waves' },
                  { name: 'Layered Cut', description: 'Versatile layers that add movement' }
                ].map((style, index) => (
                  <div key={index} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="w-full h-48 bg-gray-200 rounded-lg mb-4 flex items-center justify-center">
                      <span className="text-gray-500">Style Image</span>
                    </div>
                    <h3 className="font-semibold text-lg text-gray-900 mb-2">
                      {style.name}
                    </h3>
                    <p className="text-gray-600 mb-4">
                      {style.description}
                    </p>
                    <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors">
                      Try This Style
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="text-center mt-8 space-x-4">
          <button
            onClick={() => navigate('/upload')}
            className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Try Another Photo
          </button>
          <button
            onClick={() => navigate('/preferences', { 
              state: { imageFile, previewUrl, uploadResponse } 
            })}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Update Preferences
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;