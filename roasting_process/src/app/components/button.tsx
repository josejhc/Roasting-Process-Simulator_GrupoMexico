const Button = ({ onClick, className, text }: any) => {
// const Button = ({ onClick, className, text }) => {
  return (
    <button onClick={onClick} className={className}>
      {text}
    </button>
  );
};

export default Button;
