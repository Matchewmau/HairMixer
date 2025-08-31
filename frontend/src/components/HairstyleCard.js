import React from 'react';

const HairstyleCard = ({ image, title, description, category, onClick }) => {
  return (
    <div 
      className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl overflow-hidden hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300 cursor-pointer transform hover:scale-105 group"
      onClick={onClick}
    >
      <div className="relative aspect-[4/3] overflow-hidden">
        <img
          src={image}
          alt={title}
          className="w-full h-full object-cover object-center group-hover:scale-110 transition-transform duration-300"
          style={{ 
            objectPosition: 'center center',
            minHeight: '100%',
            minWidth: '100%'
          }}
        />
        <div className="absolute top-4 left-4">
          <span className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-3 py-1 rounded-full text-sm font-medium shadow-lg">
            {category}
          </span>
        </div>
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900/20 to-transparent"></div>
      </div>
      
      <div className="p-6">
        <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors duration-300">{title}</h3>
        <p className="text-gray-300 mb-4 leading-relaxed">{description}</p>
        
        <button className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 ease-in-out transform hover:scale-105 shadow-lg">
          Try This Style
        </button>
      </div>
    </div>
  );
};

export default HairstyleCard;