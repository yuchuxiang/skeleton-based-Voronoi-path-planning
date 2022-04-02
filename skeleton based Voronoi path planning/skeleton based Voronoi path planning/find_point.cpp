/**
 * Code for thinning a binary image using Zhang-Suen algorithm，and then using BFS to find the path.
 *
 * Author:  ChuXiang
 * 
 */
#include <opencv2/opencv.hpp>
#include "iostream"
#include <unistd.h>
#include "string.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

std::deque<std::pair<int ,int>> point_duan;// NOLINT
std::deque<std::pair<int ,int>> point_fenzhi;// NOLINT
std::deque<std::pair<int ,int>> point_fenzhi_check;// NOLINT
cv::Point start_p,end_p;
std::deque<std::pair<int,int>> all_key_points;
std::pair<int,int> find_p_start;
std::pair<int,int> find_p_end;
void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int m, n;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (m = 1; m < img.rows-1; ++m) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(m+1);

        pDst = marker.ptr<uchar>(m);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (n = 1; n < img.cols-1; ++n) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[n+1]);
            we = me;
            me = ea;
            ea = &(pCurr[n+1]);
            sw = so;
            so = se;
            se = &(pBelow[n+1]);

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

//            int condition_1=*no + *ea + *so + *we;
//            int condition_2=*ne + *se + *sw + *nw;
//            int condition_3=*ne + *se + *we;
//            int condition_4=*ne + *so + *nw;
//            int condition_5=*ea + *sw + *nw;
//            int condition_6=*no + *se + *sw;
//            int condition_7=*ea + *so + *nw;
//            int condition_8=*no + *ea + *sw;
//            int condition_9=*no + *se + *we;
//            int condition_10=*ne + *so + *we;

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[n] = 1;

        }
    }

    img &= ~marker;
}

void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

void extra_key_points( cv::Mat& bw){
    int i,j;
    uchar *s8, *s1, *s2;    // north (pAbove)
    uchar *s7, *s0, *s3;
    uchar *s6, *s5, *s4;
    uchar *pDst;
    uchar *sAbove;//矩阵第一行的三个元素
    uchar *sCurr;
    uchar *sBelow;
    int check_1[1][2];
    //初始化行
    sAbove = NULL;
    sCurr = bw.ptr<uchar>(0);//这表示这个指针指向第一行的第一元素
    sBelow = bw.ptr<uchar>(1);//这表示这个指针指向第二行的第一元素


    for ( i = 1; i < bw.rows-1; ++i) {
        //从第二行开始，结束于倒数第二行,从上到下移动行
        sAbove=sCurr;
        sCurr=sBelow;
        sBelow=bw.ptr<uchar>(i+1);

        s1=&(sAbove[0]);
        s2=&(sAbove[1]);
        s0=&(sCurr[0]);
        s3=&(sCurr[1]);
        s5=&(sBelow[0]);
        s4=&(sBelow[1]);

        for ( j = 1; j < bw.cols-1; ++j) {
            //从左到右移动列
            s8=s1;
            s1=s2;
            s2=&(sAbove[j+1]);
            s7=s0;
            s0=s3;
            s3=&(sCurr[j+1]);
            s6=s5;
            s5=s4;
            s4=&(sBelow[j+1]);

            int s_sum= *s1 + *s2 + *s3 + *s4 + *s5 + *s6 + *s7 + *s8;
            int s_center =*s0;

            int condition_1=*s1 + *s3 + *s5 + *s7;
            int condition_1e=*s2 + *s4 + *s6 + *s8;

            int condition_2=*s2 + *s4 + *s6 + *s8;
            int condition_2e=*s1 + *s3 + *s5 + *s7;

            int condition_3=*s2 + *s4 + *s7;
            int condition_3e=*s1 + *s3 + *s5 + *s6+ *s8;

            int condition_4=*s2 + *s5 + *s8;
            int condition_4e=*s1 + *s3 + *s4 + *s6+ *s7;

            int condition_5=*s3 + *s6 + *s8;
            int condition_5e=*s1 + *s2 + *s5 + *s4+ *s7;

            int condition_6=*s1 + *s4 + *s6;
            int condition_6e=*s2 + *s3 + *s5 + *s7+ *s8;

            int condition_7=*s3 + *s5 + *s8;
            int condition_7e=*s1 + *s2 + *s4 + *s6+ *s7;

            int condition_8=*s1 + *s3 + *s6;
            int condition_8e=*s2 + *s4 + *s5 + *s7+ *s8;

            int condition_9=*s1 + *s4 + *s7;
            int condition_9e=*s2 + *s3 + *s5 + *s6+ *s8;

            int condition_10=*s2 + *s5 + *s7;
            int condition_10e=*s1 + *s3 + *s4 + *s6+ *s8;

            if (s_sum ==255 &&s_center==255){
                std::pair<int ,int> ll;
                ll.first=j;
                ll.second=i;
                point_duan.push_back(ll);
            }
            if (s_center==255&&((condition_1 == 765) || (condition_2 == 765) || (condition_3 == 765) || (condition_4 == 765) ||
                                (condition_5 == 765) || (condition_6 == 765) || (condition_7 == 765) || (condition_8 == 765) ||
                                (condition_9 == 765) || (condition_10 == 765))) {
                std::pair<int, int> pp;
                pp.first = j;
                pp.second = i;
                point_fenzhi.push_back(pp);
            }

        }
    }

}

void coutpoints(const cv::Mat& bw){
    int duan_length=point_duan.size();
    int fenzhi_length=point_fenzhi.size();
    cv::Point m;
    std::vector<int> point_c;
    int i,j;


    for ( i = 0; i < duan_length; ++i) {
        //std::cout<<"端点---"<<i<<"---输出："<<point_duan.front().first<<" j输出: "<<point_duan.front().second<<std::endl;
        CvPoint p;
        p.x=point_duan.at(i).first;
        p.y=point_duan.at(i).second;

        cv::circle(bw, p, 8, cvScalar(255, 0,0 ), 1, 1, 0);
        cv::imshow("IMG1",bw);

    }


    for ( j = 0; j <fenzhi_length-1; ++j) {
        CvPoint p1;
        CvPoint p2;

        p1.x=point_fenzhi.at(j).first;
        p1.y=point_fenzhi.at(j).second;

        p2.x=point_fenzhi.at(j+1).first;
        p2.y=point_fenzhi.at(j+1).second;

        int diff=abs(p1.x-p2.x)+abs(p1.y-p2.y);
        if(diff<5){
            if(j==fenzhi_length-2){
                point_c.push_back(j);
            }
            continue;
        } else{
            point_c.push_back(j);
            if(j==fenzhi_length-2){
                point_c.push_back(0);
            }
        }
    }
    for (int i = 0; i < point_c.size(); ++i) {
        CvPoint p;
        p.x=point_fenzhi.at(point_c.at(i)).first;
        p.y=point_fenzhi.at(point_c.at(i)).second;

        std::pair<int,int> jj;
        jj.first=p.x;
        jj.second=p.y;
        point_fenzhi_check.push_back(jj);

        cv::circle(bw, p, 8, cvScalar(255, 0,0 ), 1, 1, 0);
        cv::imshow("IMG1",bw);
    }
    std::cout<<"duan:"<<point_duan.size()<<std::endl;
    std::cout<<"fenzhi:"<<point_fenzhi_check.size()<<std::endl;

}

void check_nearst_points(const cv::Mat& bw){
    float distance_start=1000;
    //std::pair<int,int> find_p_start;

    float distance_end=1000;
    //std::pair<int,int> find_p_end;

    for (int i = 0; i < bw.rows; ++i) {
        for (int j = 0; j < bw.cols; ++j) {
            if(bw.ptr<uchar>(i)[j]==255){
                float distance_start_lingshi=abs(start_p.x-j)+ abs(start_p.y-i);
                float distance_end_lingshi=abs(end_p.x-j)+ abs(end_p.y-i);

                if (distance_start>distance_start_lingshi){
                    distance_start=distance_start_lingshi;
                    find_p_start.first=j;
                    find_p_start.second=i;
                }
                if (distance_end>distance_end_lingshi){
                    distance_end=distance_end_lingshi;
                    find_p_end.first=j;
                    find_p_end.second=i;
                }

            }else{
                continue;
            }

        }
    }
    std::cout<<"sssss"<<std::endl;
    std::cout<<find_p_start.first<<"------"<<find_p_start.second<<std::endl;
    std::cout<<find_p_end.first<<"-----"<<find_p_end.second<<std::endl;
    cv::Point p_st,p_ed;
    p_st.x=find_p_start.first;
    p_st.y=find_p_start.second;

    all_key_points.push_back(find_p_start);

    p_ed.x=find_p_end.first;
    p_ed.y=find_p_end.second;

    all_key_points.push_back(find_p_end);

    cv::circle(bw, p_st, 5, cvScalar(255, 100,50 ), 1, 1, 0);
    cv::circle(bw, p_ed, 5, cvScalar(255, 100,50 ), 1, 1, 0);

    cv::imshow("IMG1",bw);
}

void push_all_points(){
    for (int i = 0; i < point_duan.size(); ++i) {
        all_key_points.push_back(point_duan.at(i));
    }
    for (int j = 0; j < point_fenzhi_check.size(); ++j) {
        all_key_points.push_back(point_fenzhi_check.at(j));
    }
}

std::deque<std::pair<int,int>> eight_neibor(int row,int col ,const cv::Mat& bw){
    //检查目标点八领域的可行点，返回其数量 和 相对位置
    std::deque<std::pair<int,int>> eight_nei;
    if(bw.at<uchar>(row-1,col)!=0){
        //std::cout<<"1 :"<<int(bw.at<uchar>(row-1,col))<<std::endl;
        std::pair<int,int> eight_1;
        eight_1.second=row-1;
        eight_1.first=col;
        eight_nei.push_back(eight_1);
    }
    if(bw.at<uchar>(row-1,col+1)!=0){
        //std::cout<<"2 :"<<int(bw.at<uchar>(row-1,col+1))<<std::endl;
        std::pair<int,int> eight_2;
        eight_2.second=row-1;
        eight_2.first=col+1;
        eight_nei.push_back(eight_2);
    }
    if(bw.at<uchar>(row,col+1)!=0){
        //std::cout<<"3 :"<<int(bw.at<uchar>(row,col+1))<<std::endl;
        std::pair<int,int> eight_3;
        eight_3.second=row;
        eight_3.first=col+1;
        eight_nei.push_back(eight_3);
    }
    if(bw.at<uchar>(row+1,col+1)!=0){
        //std::cout<<"4 :"<<int(bw.at<uchar>(row+1,col+1))<<std::endl;
        std::pair<int,int> eight_4;
        eight_4.second=row+1;
        eight_4.first=col+1;
        eight_nei.push_back(eight_4);
    }
    if(bw.at<uchar>(row+1,col)!=0){
        //std::cout<<"5 :"<<int(bw.at<uchar>(row+1,col))<<std::endl;
        std::pair<int,int> eight_5;
        eight_5.second=row+1;
        eight_5.first=col;
        eight_nei.push_back(eight_5);
    }
    if(bw.at<uchar>(row+1,col-1)!=0){
        //std::cout<<"6 :"<<int(bw.at<uchar>(row+1,col-1))<<std::endl;
        std::pair<int,int> eight_6;
        eight_6.second=row+1;
        eight_6.first=col-1;
        eight_nei.push_back(eight_6);
    }
    if(bw.at<uchar>(row,col-1)!=0){
        //std::cout<<"7 :"<<int(bw.at<uchar>(row,col-1))<<std::endl;
        std::pair<int,int> eight_7;
        eight_7.second=row;
        eight_7.first=col-1;
        eight_nei.push_back(eight_7);
    }

    if(bw.at<uchar>(row-1,col-1)!=0){
        //std::cout<<"8 :"<<int(bw.at<uchar>(row-1,col+1))<<std::endl;
        std::pair<int,int> eight_8;
        eight_8.second=row-1;
        eight_8.first=col-1;
        eight_nei.push_back(eight_8);
    }
    return eight_nei;//返回队列

}

struct CFE{
    std::pair<std::pair<int,int>,std::pair<int,int>> curpoint_fatherpoint;
    //int every_length;
};
struct CFE_ALL{
    std::pair<std::pair<int,int>,std::pair<int,int>> curpoint_fatherpoint_all;
    int every_length;
};
std::deque<std::pair<int,int>> link_path(cv::Point p1,cv::Point p2){
    std::pair<int,int> link_2points_path;
    std::deque<std::pair<int,int>> link_path_que;
    int index_xy=0;
    float s_x=fabs(p1.x-p2.x);
    float s_y=fabs(p1.y-p2.y);
    float s_max;
    if(s_x>s_y){
        s_max=s_x;
        index_xy=1;//index==1 max=x
    }else{
        s_max=s_y;
        index_xy=1;// index==2,max=y
    }
    s_max= round(s_max);
    if(s_max==0){
        link_2points_path.first=p1.x;
        link_2points_path.second=p1.y;
        link_path_que.push_back(link_2points_path);
        return link_path_que;
    }
    if (index_xy==1){
        float inter_x,inter_y;
        inter_x=(p1.x-p2.x)/s_max;
        inter_y=(p1.y-p2.y)/s_max;
        for (int i = 0; i < s_max-1; ++i) {
            float gb_x=i*inter_x;
            float gb_y=i*inter_y;
            link_2points_path.first=int(p1.x-gb_x);
            link_2points_path.second=int(p1.y-gb_y);
            link_path_que.push_back(link_2points_path);
        }
    }
    return link_path_que;
}

void find_way(std::deque<std::pair<int,int>> & all_points ,const cv::Mat& bw, const cv::Mat src){
    int all_point_size = all_points.size();
    int start_x=all_points.front().first;
    int start_y=all_points.front().second;

    int end_x=all_points.at(1).first;
    int end_y=all_points.at(1).second;
    std::pair<int,int> end_pooo;
    end_pooo.first=end_x;
    end_pooo.second=end_y;

    std::deque<std::pair<int,int>> d1;
    std::deque<CFE> d2;
    std::deque<int> d2_l;
    std::deque<CFE_ALL> d2_all;
    std::deque<std::pair<int,int>> d3;////存储路径的队列
    CFE cfe;
    CFE_ALL cfe_all;
    std::pair<std::pair<int,int>,std::pair<int,int>> curpoint_fatherpoint;
    std::deque<std::pair<int,int>> eight_find;
    std::deque<std::pair<int,int>> eight_find_all_same_layel;

    curpoint_fatherpoint.first=all_points.front();
    curpoint_fatherpoint.second=all_points.front();
    cfe.curpoint_fatherpoint=curpoint_fatherpoint;


    d1.push_back(all_points.front());
    d2.push_back(cfe);
    d2_l.push_back(1);
    cv::Mat bw_t = bw.clone();

    bw_t.at<uchar>(d1.front().second, d1.front().first) = 0;

    int count=0;
    int label=0;
    int find_first_point_success=0;

    while (d1.size()>0){

        if(label==1)
            break;
        std::pair<int,int> father_queu;
        std::pair<int,int> son_queu;
        std::pair<std::pair<int,int>,std::pair<int,int>> c_f;
        int big_length;


        for (int i = 0; i < d1.size(); ++i) {
            int x=d1.at(i).second;
            int y=d1.at(i).first;

            father_queu.first=d1.at(i).first;
            father_queu.second=d1.at(i).second;

            eight_find= eight_neibor(x,y,bw_t);

            for (int j = 0; j < eight_find.size(); ++j) {
                eight_find_all_same_layel.push_back(eight_find.at(j));
                bw_t.at<uchar>(eight_find.at(j).second, eight_find.at(j).first) = 0;
                son_queu.first=eight_find.at(j).first;
                son_queu.second=eight_find.at(j).second;
                c_f.first=son_queu;
                c_f.second=father_queu;
                cfe.curpoint_fatherpoint=c_f;
                d2.push_back(cfe);
            }
            eight_find.clear();
        }

        big_length=eight_find_all_same_layel.size();

        eight_find.clear();
        d1.clear();
        count=count+1;

        for (int k = 0; k < eight_find_all_same_layel.size(); ++k) {
            d1.push_back(eight_find_all_same_layel.at(k));
            d2_l.push_back(eight_find_all_same_layel.size());
        }
        eight_find_all_same_layel.clear();

        for (int i = 0; i < d1.size(); ++i) {
            if(d1.at(i).first==end_x && d1.at(i).second==end_y)
                label=1;
        }

    }

    std::cout<<"the length of the path is "<<count<<std::endl;
    std::cout<<"the length of the d2 is "<<d2.size()<<std::endl;
    std::cout<<"the length of the d2_l is "<<d2_l.size()<<std::endl;

    for (int i = 0; i < d2.size(); ++i) {

        std::pair<std::pair<int,int>,std::pair<int,int>> pp;
        pp.first=d2.at(i).curpoint_fatherpoint.first;
        pp.second=d2.at(i).curpoint_fatherpoint.second;
        cfe_all.curpoint_fatherpoint_all=pp;
        cfe_all.every_length=d2_l.at(i);
        d2_all.push_back(cfe_all);
    }
    std::cout<<"the length of the d2_all is "<<d2_all.size()<<std::endl;

    while (!d2_all.empty()){
        int l=d2_all.back().every_length;
        for (int i = 0; i <l ; i++) {
            if(d2_all.back().curpoint_fatherpoint_all.first==end_pooo){
                d3.push_back(end_pooo);
                end_pooo=d2_all.back().curpoint_fatherpoint_all.second;
            }
            d2_all.pop_back();
        }

    }

    cv::Mat src_t=src.clone();
    std::cout<<"the length of the d3 is "<<d3.size()<<std::endl;
    d3.pop_back();
    ///end points to skeletonq
    cv::Point nearest_end;
    nearest_end.x=find_p_end.first;
    nearest_end.y=find_p_end.second;
    std::deque<std::pair<int,int>> end_path_que;
    end_path_que=link_path(end_p,nearest_end);
    std::cout<<"the length of the end_path_que is "<<end_path_que.size()<<std::endl;
    for (int i = end_path_que.size()-1; i >=0; --i) {
        //std::cout<<end_path_que.at(i).first<<"xxvvvvxx"<<end_path_que.at(i).second<< std::endl;
        d3.push_front(end_path_que.at(i));
    }

    ////skeleton to start points
    cv::Point nearest_start;
    nearest_start.x=find_p_start.first;
    nearest_start.y=find_p_start.second;
    std::deque<std::pair<int,int>> start_path_que;
    start_path_que=link_path(nearest_start,start_p);
    std::cout<<"the length of the start_path_que is "<<start_path_que.size()<<std::endl;

    for (int i = 0; i < start_path_que.size(); ++i) {
        //std::cout<<start_path_que.at(i).first<<"xxaaaxx"<<start_path_que.at(i).second<< std::endl;
        d3.push_back(start_path_que.at(i));

    }

    for (int i = 0; i < d3.size(); ++i) {
        std::cout<<d3.at(i).first<<"xxxx"<<d3.at(i).second<< std::endl;
        cv::Point P;
        P.y=d3.at(i).second;
        P.x=d3.at(i).first;
        cv::circle(src_t,P,1,(0,255,0),-1);
    }
    cv::imshow("lujing",src_t);
    cv::waitKey();

}

void save_mat(cv::Mat & a){
    FILE *P;
    P= fopen("mat.txt","wt");
    int i,j;
    for ( i = 0; i < a.rows; ++i) {
        for ( j = 0; j < a.cols; ++j) {
            fprintf(P,"%d ",a.at<uchar>(i,j));

        }
        fprintf(P,"\n");
    }
    fclose(P);
}


/**
 * This is an example on how to call the thinning funciton above
 */



int main()
{
    cv::Mat src = cv::imread("/home/ycx/image.png");

    start_p.x=50;
    start_p.y=400;//start point ,you can set your start point here

    end_p.x=650;
    end_p.y=50;//ending point ,you can set it as well

    if (src.empty()) {
        std::cout << "000" << std::endl;
        return -1;
    }
    point_duan.clear();
    point_fenzhi.clear();
    cv::Mat bw;
    cv::Mat cw;
    cv::cvtColor(src, bw, CV_BGR2GRAY);
    cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

    thinning(bw, bw);
    cw=bw.clone();
    std::cout<<"channel:"<<bw.channels()<< std::endl;
    std::cout<<"type:"<<bw.type()<< std::endl;
    save_mat(bw);

    cv::imshow("IMGp", src);
    cv::imshow("IMG1", bw);//白色是255，黑色是0
    extra_key_points(bw);
    coutpoints(bw);
    cv::waitKey();

    check_nearst_points(bw);
    push_all_points();

    find_way(all_key_points,cw,src);
    cv::circle(bw, start_p, 5, cvScalar(255, 100,50 ), -1, 1, 0);
    cv::circle(bw, end_p, 5, cvScalar(255, 100,50 ), -1, 1, 0);
    cv::imshow("IMG1",bw);
    cv::waitKey();
    return 0;
}

