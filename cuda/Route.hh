// Number of destinations
const int DESTINATIONS = 8;

// Gives the distance between two locations (replicates problem input)
__device__ int leg_distance(int src, int dest);

// Class to hold a particular route
class Route
{
public:
    // Ordered destination list
    int route[DESTINATIONS];

    // Convert an integer to a particular route
    __device__ Route(int N)
    {
        for (int i = 0; i < DESTINATIONS; ++i)
        {
            route[i] = N % DESTINATIONS;
            N /= DESTINATIONS;
        }
    }

    // Check if the route is valid (each location exactly once)
    __device__ bool valid() const
    {
        for (int i = 0; i < DESTINATIONS; ++i)
        {
            for (int j = i + 1; j < DESTINATIONS; ++j)
            {
                if (route[i] == route[j])
                {
                    return false;
                }
            }
            if (route[i] < 0 || route[i] >= DESTINATIONS)
            {
                return false;
            }
        }
        return true;
    }

    // Calculate the distance for this route
    __device__ int distance() const
    {
        int distance = 0;
        for (int i = 0; i < DESTINATIONS - 1; ++i)
        {
            distance += leg_distance(route[i], route[i+1]);
        }
        return distance;
    }
};


// Gives the distance between two locations (replicates problem input)
__device__ int leg_distance(int src, int dest)
{
    if (src == dest || src < 0 || src > 7 || dest < 0 || dest > 7)
    {
        return -1;
    }

    if (src > dest)
    {
        int tmp = src;
        src = dest;
        dest = tmp;
    }

    switch(src)
    {
        case 0:
        {
            switch(dest)
            {
                case 1:
                    return 66;
                case 2:
                    return 28;
                case 3:
                    return 60;
                case 4:
                    return 34;
                case 5:
                    return 34;
                case 6:
                    return 3;
                case 7:
                    return 108;
            }
        }
        case 1:
        {
            switch(dest)
            {
                case 2:
                    return 22;
                case 3:
                    return 12;
                case 4:
                    return 91;
                case 5:
                    return 121;
                case 6:
                    return 111;
                case 7:
                    return 71;
            }
        }
        case 2:
        {
            switch(dest)
            {
                case 3:
                    return 39;
                case 4:
                    return 113;
                case 5:
                    return 130;
                case 6:
                    return 35;
                case 7:
                    return 40;
            }
        }
        case 3:
        {
            switch(dest)
            {
                case 4:
                    return 63;
                case 5:
                    return 21;
                case 6:
                    return 57;
                case 7:
                    return 83;
            }
        }
        case 4:
        {
            switch(dest)
            {
                case 5:
                    return 9;
                case 6:
                    return 50;
                case 7:
                    return 60;
            }
        }
        case 5:
        {
            switch(dest)
            {
                case 6:
                    return 27;
                case 7:
                    return 81;
            }
        }
        case 6:
        {
            return 90;
        }
    }
    return -1;
}