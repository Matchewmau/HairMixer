import React from 'react';

const HairstyleCard = ({ image, title, description, category, onClick }) => {
  return (
    <div 
      className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/40 hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300 cursor-pointer transform hover:scale-105 group"
      onClick={onClick}
    >
      <div className="relative aspect-[4/3] overflow-hidden">
        <img
          src={image}
          alt={title}
          className="w-full h-full object-cover object-center group-hover:scale-110 transition-transform duration-500"
          style={{ 
            objectPosition: 'center center',
            minHeight: '100%',
            minWidth: '100%'
          }}
        />
        <div className="absolute top-4 left-4">
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-lg border border-blue-500/30">
            {category}
          </span>
        </div>
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900/60 via-gray-900/20 to-transparent group-hover:from-gray-900/70 transition-all duration-300"></div>
      </div>
      
      <div className="p-6">
        <h3 className="text-xl md:text-2xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">{title}</h3>
        <p className="text-gray-300 mb-6 leading-relaxed">{description}</p>
        
        <button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-2xl border border-blue-500/30">
          Try This Style
        </button>
      </div>
    </div>
  );
};

export default HairstyleCard;