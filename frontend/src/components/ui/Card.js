import React from "react";

const Card = ({ children, className = "", hover = false, ...props }) => {
  return (
    <div
      className={`
        glass-card rounded-xl overflow-hidden
        ${
          hover
            ? "hover:border-purple-500/30 hover:shadow-purple-500/10 hover:-translate-y-1"
            : ""
        }
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
};

export default Card;
