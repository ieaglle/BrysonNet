namespace BrysonNet.ActivationFunctions
{
    public class Signum
    {
        public int Calc(double input)
        {
            return input > 0 ? 1 : -1;
        }
    }
}
