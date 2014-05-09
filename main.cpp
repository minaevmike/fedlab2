//Author: Minaev Mike RK6-103 2014
//To compile g++ main.cpp -o flab2
//To compile with debug output g++ main.cpp -DDEBUG -o flab2

#include <cstdlib>
#include <vector>
#include <iostream>
#include <ctime>
#include <map>
#include <cmath>
#include <iomanip>
#include <cmath>
#include <fstream>
static long ID = 1;
static const int fieldSize = 9;
static const int inputSize = fieldSize, outputSize = fieldSize, hiddenSize = 4 * fieldSize;
static const double DELTA = 1;
class Neuron{
private:
    std::vector<Neuron *> out, in;
    //Храним мапу с весами для каждого нейрона, ключ - id нейорона с которым соеденен данный, значение - вес
    std::map<long, double> weights;
    long id;
    double value;
    bool isInput;
    double s;
    std::vector<double> inputWeights;
public:
    double getS(){
        return this->s;
    }

    void connectTo(Neuron *neuron){
        out.push_back(neuron);
        weights[neuron->getId()] = (double)rand() / RAND_MAX - 0.5;
    }
    double calcOut(std::vector<double> v){
        double result = 0;
        if(v.size() != in.size()){
            std::cout << "SOMETHING GOES WRONG" << std::endl;
        }
        for(int i = 0;i < (int)v.size(); ++i){
#ifdef DEBUG
			std::cout << "V = " << v[i] << " weight = " << in[i]->getWeight(this->getId()) << std::endl;
#endif
			result += v[i] * in[i]->getWeight(this->getId());
        }
        this->s = result;
        this->value = (double)1 / (1 + exp(-result));
        return this->value;
    }

    double getValue(){
        return this->value;
    }

    void setValue(double value){
        this->value = value;
    }

    void setId(long id){
        this->id = id;
    }
    void addIn(Neuron *neuron){
        in.push_back(neuron);
    }

    long getId() const{
        return this->id;
    }
    Neuron(){
        this->id = ID++;
        this->isInput = false;
    }
    void setInput(){
        this->isInput = true;
        for(int i = 0; i < inputSize; ++i){
            inputWeights.push_back((double)rand() / RAND_MAX);
        }
    }

    void printNeuro(long from,double weight){
        std::cout << "Neuro id = " <<  from << " connected to neuro id = " << this->id << " weight = " << weight << std::endl;
    }
    void printConnectOut(){
        for(int i = 0; i < (int)out.size(); ++i){
            out[i]->printNeuro(this->id, weights[out[i]->getId()]);
        }
    }
    double getWeight(long id){
        return this->weights[id];
    }

    void setWeight(long id, double weight){
        this->weights[id] = weight;
    }

    void printConnectIn(){
        for(int i = 0; i < (int)in.size(); ++i){
            in[i]->printNeuro(this->getId(), in[i]->getWeight(this->getId()));
        }
    }

};
class NeuronNet{
private:
    std::vector<Neuron> input, output, hidden;
    std::vector<double> inputWeight;
    void printNet(){
        for(int i = 0; i < (int)hidden.size(); ++i){
            hidden[i].printConnectOut();
        }
        std::cout << "_________________________________________________________________" << std::endl;
        
        for(int i = 0; i < (int)input.size(); ++i){
            input[i].printConnectOut();
        }
        std::cout << "_________________________________________________________________" << std::endl;

    }

public:
    NeuronNet(){
        for(int i = 0; i < inputSize; ++i){
            input.push_back(Neuron());
        }
        for(int i = 0; i < outputSize; ++i){
            output.push_back(Neuron());
        }
        for(int i = 0; i < hiddenSize; ++i){
            hidden.push_back(Neuron());
        }
        for(int i = 0; i < (int)input.size(); ++i){
            input[i].setInput();
            for(int j = 0; j < (int)hidden.size(); ++j){
                input[i].connectTo(&hidden[j]);
            }
        }
        for(int i = 0; i < (int)hidden.size(); ++i){
            for(int j = 0; j < (int)output.size(); ++j){
                hidden[i].connectTo(&output[j]);
            }
            for(int k = 0; k < (int)input.size(); ++k){
                hidden[i].addIn(&input[k]);
            }
        }
        for(int i = 0; i < (int)output.size(); ++i){
            for(int j = 0; j < (int)hidden.size(); ++j){
                output[i].addIn(&hidden[j]);
            }
        }
#ifdef DEBUG
		this->printNet();
#endif
		std::vector<std::vector<double> > s, d;
		std::ifstream teacher("teach");
		while(teacher.good()){
			std::vector<double> s1, d1;
			double t;
			for(int i = 0; i < fieldSize; ++i){
				teacher >> t;
				s1.push_back(t);
			}
			for(int i = 0; i < fieldSize; ++i){
				teacher >> t;
				d1.push_back(t);
			}
			s.push_back(s1);
			d.push_back(d1);
		}
#ifdef DEBUG
		for(int i = 0; i < s.size() - 1;++i){
			for(int j = 0; j < s[i].size(); ++j)
				std::cout << s[i][j] << " ";
			std::cout <<"\t";
			for(int j = 0; j < d[i].size(); ++j)
				std::cout << d[i][j] << " ";
			std::cout << std::endl;
		}
#endif
		long counter = 0;
		std::cout << "Start learning " << std::endl;
		while(1){
			counter ++;
			double e = -1, t;
			for(int i = 0;i < s.size() - 1; ++i){
				teach(s[i], d[i]);
				t = error(d[i], calcOutput(s[i]));
				if (t > e)
					e = t;
			}
			if(counter%1000 == 0)
				std::cout << "Still learning" << std::endl;
			if(e < 1e-1)
				break;
		}
		std::cout << "Learning has been finished, itreations = "<< counter  << std::endl;
		std::cout << "It's time to play)" << std::endl;
		char c ='y';
		while(c == 'Y' || c == 'y'){
			play();
			std::cout << "Want to play again? [y/n]" << std::endl;
			std::cin >> c;
		}
    }
	void currentField(std::vector<double> f){
		for(int i = 0; i < f.size(); ++i){
			std::cout << std::setw(3) <<  f[i] << " ";
		}
		std::cout << std::endl;
	}
	double checkWinner(std::vector<double> f){
		if(f[0] == f[4] && f[0] == f[8])
			return f[0];
		if(f[2] == f[4] && f[2] == f[6])
			return f[2];
		for(int i = 0; i < 3; ++i){
			if(f[i * 3] == f[i * 3 + 1] && f[i * 3] == f[i * 3 + 2])
				return f[i*3];
			else
				if(f[i] == f[i + 3] && f[i] == f[i+6])
					return f[i];
		}
		return 0.5;
	}
	void play(){
		std::cout << "Info:\n \t0 - 'O' ceil\n\t0.5 - blank ceil \n\t 1 - 'X' ceil\n";
		std::vector<double> field, aiMove;
		for(int i = 0; i < fieldSize; ++i){
			field.push_back(0.5);
		}
		int moves = 0, move;
		while(moves < fieldSize){
			std::cout << "Please enter a number of ceil where you want to place 'X'" << std::endl;
			currentField(field);
			std::cin >> move;
			if(fabs(field[move - 1] - 0.5) > 1e-4){
				std::cout << "This ceil already busy, please try again" << std::endl;
				continue;
			}
			else{
				field[move - 1] = 1;
			}
			if(checkWinner(field) == 1){
				std::cout << "You win!" << std::endl;
				return;
			}
			moves++;
			aiMove = calcOutput(field);
			moves++;
			int movePos;
			double value = 1;
			currentField(aiMove);
			for(int i = 0; i < aiMove.size(); ++i){
#ifdef DEBUG
				std::cout << "AI " << aiMove[i] << " V " << value << std::endl;
#endif
				if (aiMove[i] < value && fabs(field[i] - 0.5) < 1e-4){
					value = aiMove[i];
					movePos = i;
				}
			}
			field[movePos] = 0;
			if(checkWinner(field) == 0){
				std::cout << "AI WINS!" << std::endl;
				return;
			}
			currentField(field);
		}
	}
	double error(std::vector<double> a, std::vector<double> b){
		if(a.size() != b.size()){
			std::cout<<"You idiot\n";
			return -1;
		}
		double sum = 0;
		for(int i = 0; i < (int) a.size(); ++i){
			sum+= pow(a[i] - b[i],2);
		}
		return sqrt(sum);
	}
    std::vector<double> calcOutput(std::vector<double> s){
        for(int i = 0; i < (int)s.size(); ++i){
            input[i].setValue(s[i]);
        }
        std::vector<double> hid, out;
        //Значение выходов скрытого слоя
        for(int i = 0; i < (int) hidden.size(); ++i){
            hid.push_back(hidden[i].calcOut(s));
#ifdef DEBUG
			std::cout << "S = " << hidden[i].getS() << std::endl;
			std::cout << "Value = " << hidden[i].getValue() << std::endl;
#endif
		}
        //Значения выходов выходного слоя
        for(int i = 0; i < (int) output.size(); ++i){
            out.push_back(output[i].calcOut(hid));
        }
#ifdef DEBUG
		std::cout << "In function "; 
		for(int i = 0; i < out.size(); ++i){
			std::cout << out[i] << " ";
		}
		std::cout << std::endl;
#endif
        return out;
    }

    void teach(std::vector<double> s, std::vector<double> d){
#ifdef DEBUG
		for(int i = 0; i < hidden.size();++i){
            std::cout << input[0].getWeight(hidden[i].getId()) << std::endl;
        }
        std::cout << "_________________\n";
#endif
        //Значение входного слоя
        for(int i = 0; i < (int)s.size(); ++i){
            input[i].setValue(s[i]);
        }
        std::vector<double> hiddenOut, outputOut,delta;
        //Значение выходов скрытого слоя
        for(int i = 0; i < (int) hidden.size(); ++i){
            hiddenOut.push_back(hidden[i].calcOut(s));
        }
        //Значения выходов выходного слоя
        for(int i = 0; i < (int) output.size(); ++i){
            outputOut.push_back(output[i].calcOut(hiddenOut));
        }
        //Подсчет ошибки дельта
        for(int i = 0; i < (int)output.size(); ++i){
            delta.push_back(outputOut[i] * (1 - outputOut[i]) * (d[i] - outputOut[i]));
        }
        //Изменение весов входящих в выходной
        for(int i = 0; i < (int)hidden.size(); ++i){
            for(int j = 0; j < (int)delta.size(); ++j){
                double w = hidden[i].getWeight(output[j].getId());
                double wNext = w + DELTA * delta[j] * hiddenOut[i];
#ifdef DEBUG
				std::cout << "Old weight of " << hidden[i].getId() << " - " << output[j].getId() << " is " << w << std::endl << "New weight is " << wNext << std::endl;;
#endif
				hidden[i].setWeight(output[j].getId(), wNext);
            }
        }
        //Меняем остальные веса
        for(int i = 0; i < (int)input.size();++i){
            for(int j = 0; j < (int)hidden.size();++j){
				//TODO refator to calculate delta once
                double sum = 0;
                for(int k = 0; k < (int)output.size(); ++k){
                    sum += delta[k] * hidden[j].getWeight(output[k].getId());
                }
                double delta = hiddenOut[j] * (1 - hiddenOut[j])* sum;
                double w = input[i].getWeight(hidden[j].getId());
                double wNext = w + DELTA * delta * input[i].getValue();
                input[i].setWeight(hidden[j].getId(), wNext);
            }
        }
#ifdef DEBUG
        for(int i = 0; i < hidden.size();++i){
            std::cout << input[0].getWeight(hidden[i].getId()) << std::endl;
        }
        std::cout << "_________________\n";
        for(int i = 0;i < (int)hidden.size(); ++i){
            std::cout << std::setprecision(14) <<"Out of hidden " << i << " is " << hidden[i].getValue() << std::endl;
        }
        for(int i = 0;i < (int)output.size(); ++i){
            std::cout << std::setprecision(14)<< "Out of output " << i << " is " << output[i].getValue() << std::endl;
        }
#endif

    }
};

int main()
{
    srand(time(NULL));
    std::ios_base::sync_with_stdio(false);
    NeuronNet *neuroNet = new NeuronNet();
}
